# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
import json
import threading
import queue
import glob
from functools import partial

import numpy as np
import torch
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from nemo import lightning as nl
from nemo.lightning.megatron_parallel import MegatronParallel
from transformers import T5EncoderModel, T5TokenizerFast

MegatronParallel.init_ddp = lambda self: None
from nemo.collections.diffusion.mcore_parallel_utils import Utils
from nemo.collections.diffusion.sampler.conditioner import VideoConditioner, VideoExtendConditioner
from nemo.collections.diffusion.sampler.conditioner_configs import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from nemo.collections.diffusion.sampler.cosmos.cosmos_diffusion_pipeline import CosmosDiffusionPipeline
from nemo.collections.diffusion.sampler.cosmos.cosmos_extended_diffusion_pipeline import ExtendedDiffusionPipeline
from cosmos1.models.diffusion.conditioner import DataType
from cosmos1.models.diffusion.inference.inference_utils import read_video_or_image_into_frames_BCTHW
from cosmos1.models.diffusion.nemo.inference.inference_utils_no_guardrail import process_prompt, save_video
from cosmos1.utils import log

# Global variables to store preloaded models
preloaded_models = {
    'text2world': {
        '7B': None,
        '14B': None
    },
    'video2world': {
        '7B': None,
        '14B': None
    }
}
preloaded_tokenizers = {}
preloaded_t5 = {
    'tokenizer': None,
    'model': None
}
preloaded_diffusion_pipelines = {}

# Job queue
job_queue = queue.Queue()
running = True

def print_rank_0(string: str):
    """Print only from rank 0 process."""
    try:
        rank = torch.distributed.get_rank()
        if rank == 0:
            log.info(string)
    except:
        # If distributed not initialized, just print
        log.info(string)

@torch.no_grad()
def encode_text_t5(text, max_length=512):
    """Encode text using preloaded T5 model."""
    if preloaded_t5['tokenizer'] is None or preloaded_t5['model'] is None:
        raise ValueError("T5 model and tokenizer are not preloaded!")
    
    # Ensure consistent device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    tokenizer = preloaded_t5['tokenizer']
    text_encoder = preloaded_t5['model'].to(device)
    
    batch_encoding = tokenizer.batch_encode_plus(
        [text],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )
    
    input_ids = batch_encoding.input_ids.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)
    
    outputs = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state
    
    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id]:] = 0
    
    return encoded_text

def create_condition_latent_from_input_frames(tokenizer, input_path, height, width, num_frames_condition=1):
    """Create latent representation from input frames."""
    # Ensure consistent device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Read input frames
    input_path_format = input_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_path,
        input_path_format=input_path_format,
        H=height,
        W=width,
    )
    
    # Move to the correct device
    input_frames = input_frames.to(device)
    
    # Handle the case where we need more frames than provided
    B, C, T, H, W = input_frames.shape
    if T < num_frames_condition:
        pad_frames = input_frames[:, :, -1:].repeat(1, 1, num_frames_condition - T, 1, 1)
        input_frames = torch.cat([input_frames, pad_frames], dim=2)
    
    num_frames_encode = tokenizer.pixel_chunk_duration
    
    # Put the conditional frames at the beginning of the video
    condition_frames = input_frames[:, :, -num_frames_condition:]
    padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
    encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2).to(device)
    
    # Ensure tokenizer is on the same device
    tokenizer = tokenizer.to(device)
    latent = tokenizer.encode(encode_input_frames)
    
    return latent, encode_input_frames

def compute_num_latent_frames(tokenizer, num_input_frames: int, downsample_factor=8) -> int:
    """Compute number of latent frames."""
    num_latent_frames = (
        num_input_frames // tokenizer.video_vae.pixel_chunk_duration * tokenizer.video_vae.latent_chunk_duration
    )
    if num_input_frames % tokenizer.video_vae.latent_chunk_duration == 1:
        num_latent_frames += 1
    elif num_input_frames % tokenizer.video_vae.latent_chunk_duration > 1:
        assert (
            num_input_frames % tokenizer.video_vae.pixel_chunk_duration - 1
        ) % downsample_factor == 0, (
            f"num_input_frames % tokenizer.video_vae.pixel_chunk_duration - 1 must be divisible by {downsample_factor}"
        )
        num_latent_frames += 1 + (num_input_frames % tokenizer.video_vae.pixel_chunk_duration - 1) // downsample_factor

    return num_latent_frames

def preload_models(args):
    """Preload all required models into memory."""
    print_rank_0("Preloading models...")
    
    # Ensure we're using device 0 for consistency across all operations
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print_rank_0(f"Using primary device: {device}")
    
    # Initialize distributed environment
    Utils.initialize_distributed(1, 1, context_parallel_size=args.cp_size)
    model_parallel_cuda_manual_seed(args.seed)
    
    # Initialize T5 model and tokenizer
    print_rank_0("Loading T5 model and tokenizer...")
    
    # Handle potential wildcard in T5 cache path
    t5_cache_dir = args.t5_cache_dir
    if '*' in t5_cache_dir:
        t5_dirs = glob.glob(t5_cache_dir)
        if not t5_dirs:
            raise ValueError(f"No T5 directories found matching pattern: {t5_cache_dir}")
        t5_cache_dir = t5_dirs[0]
        print_rank_0(f"Using T5 cache directory: {t5_cache_dir}")
    
    preloaded_t5['tokenizer'] = T5TokenizerFast.from_pretrained("google-t5/t5-11b", cache_dir=t5_cache_dir)
    
    # Load T5 model with lower precision to save memory
    preloaded_t5['model'] = T5EncoderModel.from_pretrained(
        "google-t5/t5-11b", 
        cache_dir=t5_cache_dir,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory savings
        device_map="auto"  # Allow model to be split across devices if needed
    )
    preloaded_t5['model'].eval()
    
    # Initialize video tokenizer
    print_rank_0("Loading video tokenizer...")
    from nemo.collections.diffusion.models.model import DiT7BConfig
    
    # Find the actual snapshot directory by expanding the wildcard
    tokenizer_pattern = os.path.join(args.checkpoints_root, args.tokenizer_dir)
    tokenizer_dirs = glob.glob(tokenizer_pattern)
    if not tokenizer_dirs:
        raise ValueError(f"No directories found matching pattern: {tokenizer_pattern}")
    vae_path = tokenizer_dirs[0]  # Use the first matching directory
    print_rank_0(f"Using tokenizer path: {vae_path}")
    
    dit_config = DiT7BConfig(vae_path=vae_path)
    tokenizer = dit_config.configure_vae()
    preloaded_tokenizers['tokenizer'] = tokenizer
    
    # Initialize diffusion models
    for model_size in ['7B']:
        if args.preload_text2world:
            print_rank_0(f"Loading Text2World {model_size} model...")
            # Initialize DiT model
            from nemo.collections.diffusion.models.model import DiT7BConfig, DiTModel
            
            dit_config = DiT7BConfig()
            dit_model = DiTModel(dit_config)
            
            # Initialize model parallel strategy
            strategy = nl.MegatronStrategy(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=args.cp_size,
                pipeline_dtype=torch.bfloat16,
            )
            
            # Initialize trainer
            trainer = nl.Trainer(
                devices=args.num_devices,
                max_steps=1,
                accelerator="gpu",
                strategy=strategy,
                plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            )
            
            # Convert trainer to fabric for inference
            fabric = trainer.to_fabric()
            fabric.strategy.checkpoint_io.save_ckpt_format = "zarr"
            fabric.strategy.checkpoint_io.validate_access_integrity = False
            fabric.strategy.checkpoint_io.load_checkpoint = partial(fabric.strategy.checkpoint_io.load_checkpoint, strict=False)
            StrictHandling.requires_global_app_metadata = staticmethod(lambda val: False)
            
            # Load model checkpoint
            nemo_checkpoint_pattern = os.path.join(args.checkpoints_root, f"models--nvidia--Cosmos-1.0-Diffusion-{model_size}-Text2World/snapshots/*/nemo")
            nemo_checkpoint_dirs = glob.glob(nemo_checkpoint_pattern)
            if not nemo_checkpoint_dirs:
                raise ValueError(f"No directories found matching pattern: {nemo_checkpoint_pattern}")
            nemo_checkpoint = nemo_checkpoint_dirs[0]  # Use the first matching directory
            print_rank_0(f"Using Text2World model checkpoint: {nemo_checkpoint}")
            
            # Make sure to load model to device 0
            device = torch.device('cuda:0')
            model = fabric.load_model(nemo_checkpoint, dit_model).to(device=device, dtype=torch.bfloat16)
            
            # Extra insurance that the model is on the correct device
            for param in model.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
            
            # Set up diffusion pipeline
            conditioner = VideoConditioner(
                text=TextConfig(),
                fps=FPSConfig(),
                num_frames=NumFramesConfig(),
                image_size=ImageSizeConfig(),
                padding_mask=PaddingMaskConfig(),
            )
            
            diffusion_pipeline = CosmosDiffusionPipeline(
                net=model.module, conditioner=conditioner, sampler_type="RES", seed=args.seed
            )
            
            preloaded_models['text2world'][model_size] = model
            preloaded_diffusion_pipelines[f'text2world_{model_size}'] = diffusion_pipeline
        
        if args.preload_video2world:
            print_rank_0(f"Loading Video2World {model_size} model...")
            # Initialize DiT model for Video2World
            from nemo.collections.diffusion.models.model import DiT7BVideo2WorldConfig, DiTModel
            
            dit_config = DiT7BVideo2WorldConfig()
            dit_model = DiTModel(dit_config)
            
            # Initialize model parallel strategy
            strategy = nl.MegatronStrategy(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=args.cp_size,
                pipeline_dtype=torch.bfloat16,
            )
            
            # Initialize trainer
            trainer = nl.Trainer(
                devices=args.num_devices,
                max_steps=1,
                accelerator="gpu",
                strategy=strategy,
                plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            )
            
            # Convert trainer to fabric for inference
            fabric = trainer.to_fabric()
            fabric.strategy.checkpoint_io.save_ckpt_format = "zarr"
            fabric.strategy.checkpoint_io.validate_access_integrity = False
            fabric.strategy.checkpoint_io.load_checkpoint = partial(fabric.strategy.checkpoint_io.load_checkpoint, strict=False)
            StrictHandling.requires_global_app_metadata = staticmethod(lambda val: False)
            
            # Load model checkpoint
            nemo_checkpoint_pattern = os.path.join(args.checkpoints_root, f"models--nvidia--Cosmos-1.0-Diffusion-{model_size}-Video2World/snapshots/*/nemo")
            nemo_checkpoint_dirs = glob.glob(nemo_checkpoint_pattern)
            if not nemo_checkpoint_dirs:
                raise ValueError(f"No directories found matching pattern: {nemo_checkpoint_pattern}")
            nemo_checkpoint = nemo_checkpoint_dirs[0]  # Use the first matching directory
            print_rank_0(f"Using Video2World model checkpoint: {nemo_checkpoint}")
            
            # Make sure to load model to device 0
            device = torch.device('cuda:0')
            model = fabric.load_model(nemo_checkpoint, dit_model).to(device=device, dtype=torch.bfloat16)
            
            # Extra insurance that the model is on the correct device
            for param in model.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
            
            # Set up diffusion pipeline
            conditioner = VideoExtendConditioner(
                text=TextConfig(),
                fps=FPSConfig(),
                num_frames=NumFramesConfig(),
                image_size=ImageSizeConfig(),
                padding_mask=PaddingMaskConfig(),
                video_cond_bool=VideoCondBoolConfig(),
            )
            
            diffusion_pipeline = ExtendedDiffusionPipeline(
                net=model.module, conditioner=conditioner, sampler_type="RES", seed=args.seed
            )
            diffusion_pipeline.conditioner.data_type = DataType.VIDEO
            
            preloaded_models['video2world'][model_size] = model
            preloaded_diffusion_pipelines[f'video2world_{model_size}'] = diffusion_pipeline
    
    print_rank_0("Model preloading complete!")

def run_text2world_inference(job_data):
    """Run text2world inference with preloaded models."""
    print_rank_0(f"Running Text2World inference for job: {job_data['id']}")
    
    # Extract job parameters
    prompt = job_data['prompt']
    model_size = job_data.get('model_size', '7B')
    guidance = job_data.get('guidance', 7.0)
    num_steps = job_data.get('num_steps', 35)
    video_frames = job_data.get('video_frames', 121)
    height = job_data.get('height', 704)
    width = job_data.get('width', 1280)
    fps = job_data.get('fps', 24)
    output_path = job_data['output_path']
    
    # Ensure we're operating on the first GPU (assuming rank 0 is on cuda:0)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Get preloaded models
    tokenizer = preloaded_tokenizers['tokenizer']
    # Make sure tokenizer is on the right device
    tokenizer = tokenizer.to(device)
    diffusion_pipeline = preloaded_diffusion_pipelines[f'text2world_{model_size}']
    
    # Skip safety checks
    print_rank_0(f"Processing prompt: {prompt}")
    
    # Encode text
    # Encode text to T5 embedding
    t5_embedding_max_length = 512
    out = encode_text_t5(prompt)[0]
    encoded_text = torch.tensor(out, dtype=torch.bfloat16, device=device)
    
    # Padding T5 embedding to t5_embedding_max_length
    L, C = encoded_text.shape
    t5_embed = torch.zeros(1, t5_embedding_max_length, C, dtype=torch.bfloat16, device=device)
    t5_embed[0, :L] = encoded_text
    
    # Prepare data batch
    t, h, w = video_frames, height, width
    state_shape = [
        tokenizer.channel,
        tokenizer.get_latent_num_frames(t),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    
    data_batch = {
        "video": torch.zeros((1, 3, t, h, w), dtype=torch.uint8, device=device),
        "t5_text_embeddings": t5_embed,
        "t5_text_mask": torch.ones(1, t5_embedding_max_length, dtype=torch.bfloat16, device=device),
        # other conditions
        "image_size": torch.tensor(
            [[height, width, height, width]] * 1, dtype=torch.bfloat16, device=device
        ),
        "fps": torch.tensor([fps] * 1, dtype=torch.bfloat16, device=device),
        "num_frames": torch.tensor([video_frames] * 1, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((1, 1, height, width), dtype=torch.bfloat16, device=device),
    }
    
    # Run inference
    data_batch["inference_fwd"] = True
    sample = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=state_shape,
        num_steps=num_steps,
        is_negative_prompt=False
    )
    
    # Post-process and save
    sigma_data = 0.5
    grid = (1.0 + tokenizer.decode(sample / sigma_data)).clamp(0, 2) / 2
    grid = (grid[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
    
    # Save without safety checks
    kwargs = {
        "fps": fps,
        "quality": 5,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{width}x{height}"],
        "output_params": ["-f", "mp4"],
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import imageio
    imageio.mimsave(output_path, grid, "mp4", **kwargs)
    
    print_rank_0(f"Saved video to {output_path}")
    
    return {
        "id": job_data['id'],
        "status": "complete",
        "output_path": output_path
    }

def run_video2world_inference(job_data):
    """Run video2world inference with preloaded models."""
    print_rank_0(f"Running Video2World inference for job: {job_data['id']}")
    
    # Extract job parameters
    prompt = job_data['prompt']
    input_path = job_data['input_path']
    num_input_frames = job_data.get('num_input_frames', 1)
    model_size = job_data.get('model_size', '7B')
    guidance = job_data.get('guidance', 7.0)
    num_steps = job_data.get('num_steps', 35)
    video_frames = job_data.get('video_frames', 121)
    height = job_data.get('height', 704)
    width = job_data.get('width', 1280)
    fps = job_data.get('fps', 24)
    output_path = job_data['output_path']
    
    # Ensure we're operating on the first GPU (assuming rank 0 is on cuda:0)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Get preloaded models
    tokenizer = preloaded_tokenizers['tokenizer']
    # Make sure tokenizer is on the right device
    tokenizer = tokenizer.to(device)
    diffusion_pipeline = preloaded_diffusion_pipelines[f'video2world_{model_size}']
    
    # Skip safety checks
    print_rank_0(f"Processing prompt: {prompt} with input: {input_path}")
    
    # Encode text
    # Encode text to T5 embedding
    t5_embedding_max_length = 512
    out = encode_text_t5(prompt)[0]
    encoded_text = torch.tensor(out, dtype=torch.bfloat16, device=device)
    
    # Padding T5 embedding to t5_embedding_max_length
    L, C = encoded_text.shape
    t5_embed = torch.zeros(1, t5_embedding_max_length, C, dtype=torch.bfloat16, device=device)
    t5_embed[0, :L] = encoded_text
    
    # Prepare data batch
    t, h, w = video_frames, height, width
    state_shape = [
        tokenizer.channel,
        tokenizer.get_latent_num_frames(t),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    
    data_batch = {
        "video": torch.zeros((1, 3, t, h, w), dtype=torch.uint8, device=device),
        "t5_text_embeddings": t5_embed,
        "t5_text_mask": torch.ones(1, t5_embedding_max_length, dtype=torch.bfloat16, device=device),
        # other conditions
        "image_size": torch.tensor(
            [[height, width, height, width]] * 1, dtype=torch.bfloat16, device=device
        ),
        "fps": torch.tensor([fps] * 1, dtype=torch.bfloat16, device=device),
        "num_frames": torch.tensor([video_frames] * 1, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros((1, 1, height, width), dtype=torch.bfloat16, device=device),
    }
    
    # Create latent conditioning from input image/video
    condition_latent, _ = create_condition_latent_from_input_frames(
        tokenizer, input_path, height, width, num_input_frames
    )
    data_batch["condition_latent"] = condition_latent
    
    # Run inference
    data_batch["inference_fwd"] = True
    augment_sigma = 0.001
    num_of_latent_condition = compute_num_latent_frames(tokenizer, num_input_frames)
    
    sample = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=state_shape,
        num_steps=num_steps,
        is_negative_prompt=False,
        num_condition_t=num_of_latent_condition,
        condition_latent=data_batch["condition_latent"],
        condition_video_augment_sigma_in_inference=augment_sigma,
    )
    
    # Post-process and save
    sigma_data = 0.5
    grid = (1.0 + tokenizer.decode(sample / sigma_data)).clamp(0, 2) / 2
    grid = (grid[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
    
    # Save without safety checks
    kwargs = {
        "fps": fps,
        "quality": 5,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{width}x{height}"],
        "output_params": ["-f", "mp4"],
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import imageio
    imageio.mimsave(output_path, grid, "mp4", **kwargs)
    
    print_rank_0(f"Saved video to {output_path}")
    
    return {
        "id": job_data['id'],
        "status": "complete",
        "output_path": output_path
    }

def worker_thread():
    """Worker thread to process jobs from the queue."""
    global running
    
    while running:
        try:
            # Check if there's a job
            try:
                job = job_queue.get(timeout=1.0)  # 1-second timeout to check running flag
            except queue.Empty:
                continue
            
            print_rank_0(f"Processing job: {job['id']}")
            
            # Track start time
            start_time = time.time()
            
            # Process job based on type
            if job['type'] == 'text2world':
                result = run_text2world_inference(job)
            elif job['type'] == 'video2world':
                result = run_video2world_inference(job)
            else:
                result = {
                    "id": job['id'],
                    "status": "failed",
                    "error": f"Unknown job type: {job['type']}"
                }
            
            # Record processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            print_rank_0(f"Completed job {job['id']} in {processing_time:.2f} seconds")
            
            # Write result to output file
            result_path = os.path.join(job['output_dir'], f"{job['id']}_result.json")
            with open(result_path, 'w') as f:
                json.dump(result, f)
            
            # Mark job as done
            job_queue.task_done()
            
        except Exception as e:
            print_rank_0(f"Error processing job: {str(e)}")
            
            try:
                # If we have job info, write error
                if 'id' in job and 'output_dir' in job:
                    error_result = {
                        "id": job['id'],
                        "status": "failed",
                        "error": str(e)
                    }
                    result_path = os.path.join(job['output_dir'], f"{job['id']}_result.json")
                    with open(result_path, 'w') as f:
                        json.dump(error_result, f)
            except:
                pass
                
            # Sleep a bit to avoid tight loop on persistent error
            time.sleep(1.0)

def start_worker():
    """Start the worker thread."""
    worker = threading.Thread(target=worker_thread)
    worker.daemon = True
    worker.start()
    return worker

def add_job(job_data):
    """Add a job to the queue."""
    job_queue.put(job_data)
    print_rank_0(f"Added job to queue: {job_data['id']}")
    return job_data['id']

def main():
    """Main function to start the server and process command line arguments."""
    parser = argparse.ArgumentParser(description="Preloaded inference server for Cosmos models")
    
    # Model loading options - we'll make these mutually exclusive
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--preload_text2world", action="store_true", help="Preload only Text2World model")
    model_group.add_argument("--preload_video2world", action="store_true", help="Preload only Video2World model")
    
    # Hardware configuration
    parser.add_argument("--num_devices", type=int, default=8, help="Number of devices for inference")
    parser.add_argument("--cp_size", type=int, default=8, help="Number of cp ranks for multi-gpu inference")
    
    # Directories
    parser.add_argument("--job_dir", type=str, default="/workspace/world-model-portal-backend/static/jobs",
                      help="Directory for job inputs and outputs")
    parser.add_argument("--checkpoints_root", type=str, default="/workspace/checkpoints/hub",
                      help="Root directory containing model checkpoints")
    parser.add_argument("--tokenizer_dir", type=str, default="models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8/snapshots/*/",
                      help="Directory for video tokenizer relative to checkpoints_root")
    parser.add_argument("--t5_cache_dir", type=str, default="/workspace/checkpoints/hub/models--google-t5--t5-11b/snapshots/*/",
                      help="Path to T5 model")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()
    
    # Create job directory if needed
    os.makedirs(args.job_dir, exist_ok=True)
    
    # Create input job monitor path
    job_input_path = os.path.join(args.job_dir, "input")
    os.makedirs(job_input_path, exist_ok=True)
    
    # Preload models
    preload_models(args)
    
    # Start worker thread
    worker = start_worker()
    
    # Watch for job files
    print_rank_0(f"Watching for job files in: {job_input_path}")
    print_rank_0("Usage: Create a JSON file in this directory with the following format:")
    print_rank_0("""
    For text2world:
    {
        "id": "unique_job_id",
        "type": "text2world",
        "prompt": "your text prompt",
        "output_path": "/path/to/output.mp4" 
    }
    
    For video2world:
    {
        "id": "unique_job_id", 
        "type": "video2world",
        "prompt": "your text prompt",
        "input_path": "/path/to/input/image_or_video.mp4",
        "num_input_frames": 1,
        "output_path": "/path/to/output.mp4"
    }
    """)
    
    try:
        while True:
            # Check for new job files
            for filename in os.listdir(job_input_path):
                if filename.endswith(".json"):
                    job_file = os.path.join(job_input_path, filename)
                    try:
                        with open(job_file, 'r') as f:
                            job_data = json.load(f)
                        
                        # Add output directory to job data
                        job_data['output_dir'] = args.job_dir
                        
                        # Add job to queue
                        add_job(job_data)
                        
                        # Move job file to prevent reprocessing
                        processed_file = os.path.join(args.job_dir, "processed", filename)
                        os.makedirs(os.path.join(args.job_dir, "processed"), exist_ok=True)
                        os.rename(job_file, processed_file)
                        
                    except Exception as e:
                        print_rank_0(f"Error processing job file {filename}: {str(e)}")
                        # Move to error directory
                        error_file = os.path.join(args.job_dir, "error", filename)
                        os.makedirs(os.path.join(args.job_dir, "error"), exist_ok=True)
                        os.rename(job_file, error_file)
            
            # Sleep to avoid high CPU usage
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print_rank_0("Shutting down...")
        global running
        running = False
        worker.join(timeout=5.0)

if __name__ == "__main__":
    main()