import os
import uuid
import asyncio
import tempfile
import zipfile
import shutil
import aiohttp
import time
import json
from typing import Dict, Optional, Tuple, Any, List

from app.core.logger import get_logger
from app.core.config import get_settings, nvidia_api_semaphore

logger = get_logger()
settings = get_settings()

class VideoService:
    """Service for handling video generation requests using Nvidia's API."""

    def __init__(self):
        """Initialize the video service with worker pool configuration."""
        settings = get_settings()
        
        # In-memory store for job statuses
        self.video_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Worker pool configuration
        self.max_concurrent_jobs = 1  # Only one concurrent job for GPU resource management
        self.active_jobs = 0          # Currently active jobs
        self.job_queue = asyncio.Queue()  # Queue for pending jobs
        self.worker_task = None
        
        # Rate limiting config
        self.retry_delay = 5
        self.max_retries = 3
        
        # Log initialization
        logger.info("Video service initialized for local Cosmos inference")
        
    async def startup(self):
        """Start the worker pool - must be called from an async context"""
        if self.worker_task is None:
            # Start worker pool
            self.worker_task = asyncio.create_task(self._job_worker())
            # Make sure worker doesn't die if there's an exception
            self.worker_task.add_done_callback(self._restart_worker)
            logger.info("Video service worker started")
        
    def _restart_worker(self, future):
        """Callback to restart the worker if it crashes"""
        if future.exception():
            logger.error(f"Worker crashed with exception: {future.exception()}")
            # Need to restart in an async context - we'll do this on the next API call
            self.worker_task = None
            logger.info("Worker task will be restarted on next API call")
            
    # Remove API key related methods as they're no longer needed for local inference
    
    async def _job_worker(self):
        """Worker that processes jobs from the queue with rate limiting"""
        logger.info("Starting video generation worker pool")
        
        # Maximum age for jobs in queue (5 minutes in seconds)
        MAX_JOB_AGE = 300  # 5 minutes
        
        while True:
            try:
                # Check if we can process more jobs
                if self.active_jobs < self.max_concurrent_jobs:
                    try:
                        # Check and clean expired jobs in queue
                        await self._clean_expired_jobs(MAX_JOB_AGE)
                        
                        # Get next job with timeout (so we can check active_jobs periodically)
                        job_id, prompt, websocket, timestamp = await asyncio.wait_for(
                            self.job_queue.get(), timeout=1.0
                        )
                        
                        # Check if job is too old
                        current_time = time.time()
                        job_age = current_time - timestamp
                        
                        if job_age > MAX_JOB_AGE:
                            # Skip this job and update its status
                            logger.warning(f"Skipping job {job_id} as it's too old ({job_age:.1f} seconds)")
                            self.video_jobs[job_id] = {
                                "status": "expired",
                                "message": f"Job expired after waiting in queue for {job_age:.1f} seconds",
                                "progress": 0,
                                "video_url": None
                            }
                            # Mark job as done in queue and continue
                            self.job_queue.task_done()
                            continue
                        
                        # Update job status
                        self.video_jobs[job_id]["status"] = "starting"
                        self.video_jobs[job_id]["message"] = "Job picked up by worker"
                        
                        # WebSocket updates removed
                        
                        # Start processing
                        self.active_jobs += 1
                        logger.debug(f"Starting job {job_id} (active jobs: {self.active_jobs}/{self.max_concurrent_jobs})")
                        
                        # Start the job processing
                        task = asyncio.create_task(
                            self._process_video_generation(job_id, prompt, websocket)
                        )
                        
                        # Add callback to decrease active job count when done
                        def on_complete(_):
                            self.active_jobs = max(0, self.active_jobs - 1)
                            logger.debug(f"Completed job {job_id} (active jobs: {self.active_jobs}/{self.max_concurrent_jobs})")
                            
                        task.add_done_callback(on_complete)
                        
                        # Mark task as done in queue
                        self.job_queue.task_done()
                        
                    except asyncio.TimeoutError:
                        # No job available, continue
                        pass
                else:
                    # At max capacity, wait before checking again
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in job worker: {e}")
                await asyncio.sleep(1)  # Avoid tight loop in case of persistent errors
                
    async def _clean_expired_jobs(self, max_age_seconds):
        """Clean up expired jobs from the queue
        
        Args:
            max_age_seconds: Maximum job age in seconds before removing from queue
        """
        # We can't directly access queue items, need to rebuild the queue
        temp_queue = asyncio.Queue()
        expired_count = 0
        
        # Try to get all items from the queue without blocking
        while True:
            try:
                job_id, prompt, websocket, timestamp = await asyncio.wait_for(
                    self.job_queue.get(), timeout=0.01
                )
                
                # Check job age
                current_time = time.time()
                job_age = current_time - timestamp
                
                if job_age > max_age_seconds:
                    # Job expired, update status
                    logger.warning(f"Removing expired job {job_id} from queue (age: {job_age:.1f} seconds)")
                    self.video_jobs[job_id] = {
                        "status": "expired",
                        "message": f"Job expired after waiting in queue for {job_age:.1f} seconds",
                        "progress": 0,
                        "video_url": None
                    }
                    expired_count += 1
                    # Job is not added back to the queue
                else:
                    # Job is still valid, add it back to the temp queue
                    await temp_queue.put((job_id, prompt, websocket, timestamp))
                
                # Mark as done in original queue
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                # No more items in the queue
                break
        
        # Now add all items from temp queue back to original queue
        while not temp_queue.empty():
            item = await temp_queue.get()
            await self.job_queue.put(item)
            temp_queue.task_done()
        
        if expired_count > 0:
            logger.info(f"Cleaned {expired_count} expired jobs from the queue")
    
    async def generate_video(self, prompt: str) -> str:
        """
        Generate a video from a prompt using Nvidia API.
        
        Args:
            prompt: The text-to-video prompt
            
        Returns:
            job_id: Unique identifier for tracking the video generation
        """
        # Ensure the worker is running
        await self.startup()
        
        job_id = str(uuid.uuid4())
        
        # Initialize job in memory store immediately
        self.video_jobs[job_id] = {
            "status": "pending",
            "message": "Job queued and will start soon",
            "progress": 0,
            "video_url": None
        }
        
        # Add job to the queue for processing by worker pool with current timestamp
        logger.debug(f"Queueing job {job_id}")
        current_time = time.time()
        await self.job_queue.put((job_id, prompt, None, current_time))  # Added timestamp
        
        # Return job_id immediately
        return job_id
        

    # Remove the _detect_input_type_and_frames method as it's no longer needed

    async def _process_video_generation(self, job_id: str, prompt: str, video_path: str = None, websocket=None) -> None:
        """
        Process the video generation job using the local Cosmos installation.
        
        Args:
            job_id: The job ID
            prompt: The text-to-video prompt
            video_path: Optional path to user's video/image for video2world
            websocket: Parameter kept for backward compatibility
        """
        try:
            # Update status in memory store
            self.video_jobs[job_id] = {
                "status": "generating",
                "message": "Generating video from prompt",
                "progress": 10,
                "video_url": None
            }
            
            # Create directory for storing the video with absolute path
            output_dir = os.path.join(os.getcwd(), f"static/videos/{job_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure the parent directory for the log file exists (might be an issue with redirecting output)
            log_dir = os.path.dirname(os.path.join(output_dir, "cosmos_log.txt"))
            os.makedirs(log_dir, exist_ok=True)
            
            # Determine number of available GPUs
            try:
                import torch
                num_devices = torch.cuda.device_count()
                logger.info(f"Detected {num_devices} GPU devices")
                if num_devices == 0:
                    num_devices = 1  # Fallback to CPU
                    logger.warning("No GPUs detected, falling back to CPU mode (will be slow)")
            except ImportError:
                logger.warning("Could not import torch to detect GPUs, defaulting to 1")
                num_devices = 1
            
            # Update job status to waiting for GPU availability
            self.video_jobs[job_id] = {
                "status": "pending",
                "message": "Waiting for GPU availability (only one job can run at a time)...",
                "progress": 5,
                "video_url": None
            }
            
            # Determine if this is a text2world or video2world job
            model_type = "text2world"
            num_input_frames = 1
            
            if video_path:
                model_type = "video2world"
                # Check file extension to determine if image or video
                file_ext = video_path.split('.')[-1].lower()
                
                # Image formats
                if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']:
                    num_input_frames = 1
                    logger.info("Using image input with 1 frame")
                # Video formats
                elif file_ext in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
                    num_input_frames = 9
                    logger.info("Using video input with 9 frames")
                else:
                    # Default to image for any other format
                    num_input_frames = 1
                    logger.warning(f"Unsupported file format: {file_ext}, treating as image with 1 frame")
            
            # We still use a semaphore to prevent multiple concurrent GPU jobs
            async with nvidia_api_semaphore:
                logger.debug(f"Acquired GPU access for job {job_id}")
                
                # Update job status to generating
                self.video_jobs[job_id] = {
                    "status": "generating",
                    "message": "Starting video generation on local GPU...",
                    "progress": 10,
                    "video_url": None
                }
                
                # Prepare output video path (absolute path)
                output_video_path = os.path.join(output_dir, "video.mp4")
                log_file_path = os.path.join(output_dir, "cosmos_log.txt")
                
                # Prepare the command to run Cosmos locally
                if model_type == "text2world":
                    cosmos_command = [
                        "bash", "-c",
                        f"cd /workspace/Cosmos && "
                        f"NVTE_FUSED_ATTN=0 "
                        f"torchrun --nproc_per_node={num_devices} "
                        f"cosmos1/models/diffusion/nemo/inference/general.py "
                        f"--model Cosmos-1.0-Diffusion-7B-Text2World "
                        f"--cp_size {num_devices} "
                        f"--num_devices {num_devices} "
                        f"--video_save_path \"{output_video_path}\" "
                        f"--guidance 7 "
                        f"--seed 1 "
                        f"--prompt \"{prompt}\" "
                        f"--guardrail_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots/*/ "
                        f"--tokenizer_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8/snapshots/*/ "
                        f"--prompt_upsampler_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World/snapshots/*/ "
                        f"--nemo_checkpoint /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World/snapshots/*/nemo "
                        f"--t5_cache_dir /workspace/checkpoints/hub/models--google-t5--t5-11b/snapshots/*/ "
                        f"--cosmos_assets_dir /workspace/checkpoints "
                        f"> \"{log_file_path}\" 2>&1"
                    ]
                else:  # video2world
                    cosmos_command = [
                        "bash", "-c",
                        f"cd /workspace/Cosmos && "
                        f"NVTE_FUSED_ATTN=0 "
                        f"torchrun --nproc_per_node={num_devices} "
                        f"cosmos1/models/diffusion/nemo/inference/video2world.py "
                        f"--model Cosmos-1.0-Diffusion-7B-Video2World "
                        f"--cp_size {num_devices} "
                        f"--num_devices {num_devices} "
                        f"--video_save_path \"{output_video_path}\" "
                        f"--guidance 7 "
                        f"--seed 1 "
                        f"--prompt \"{prompt}\" "
                        f"--conditioned_image_or_video_path \"{video_path}\" "
                        f"--num_input_frames {num_input_frames} "
                        f"--enable_prompt_upsampler "
                        f"--guardrail_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots/*/ "
                        f"--tokenizer_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8/snapshots/*/ "
                        f"--prompt_upsampler_dir /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World/snapshots/*/ "
                        f"--nemo_checkpoint /workspace/checkpoints/hub/models--nvidia--Cosmos-1.0-Diffusion-7B-Video2World/snapshots/*/nemo "
                        f"--t5_cache_dir /workspace/checkpoints/hub/models--google-t5--t5-11b/snapshots/*/ "
                        f"--cosmos_assets_dir /workspace/checkpoints "
                        f"> \"{log_file_path}\" 2>&1"
                    ]
                
                # Run the command asynchronously
                logger.info(f"Running Cosmos command for job {job_id}: {' '.join(cosmos_command)}")
                process = await asyncio.create_subprocess_exec(
                    *cosmos_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Update progress while process is running
                progress = 10
                start_time = time.time()
                
                # Poll for process completion and update progress periodically
                while True:
                    # Check if process has completed
                    if process.returncode is not None:
                        break
                    
                    # Calculate progress based on elapsed time
                    # Typical generation takes about 2-3 minutes
                    elapsed_time = time.time() - start_time
                    estimated_total_time = 180  # 3 minutes in seconds
                    progress = min(90, int(10 + (elapsed_time / estimated_total_time) * 80))
                    
                    # Read log file to estimate progress if available
                    log_file = os.path.join(output_dir, "cosmos_log.txt")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, "r") as f:
                                log_content = f.read()
                                # Check for progress indicators in the log
                                if "%" in log_content:
                                    import re
                                    # Look for patterns like "50%" or "Step 75/100"
                                    percent_matches = re.findall(r"(\d+)%", log_content)
                                    if percent_matches:
                                        # Use the last percentage mentioned
                                        log_progress = int(percent_matches[-1])
                                        progress = min(90, 10 + int(log_progress * 0.8))
                                    else:
                                        # Try step pattern
                                        step_matches = re.findall(r"Step (\d+)/(\d+)", log_content)
                                        if step_matches:
                                            current, total = map(int, step_matches[-1])
                                            progress = min(90, 10 + int(80 * current / total))
                        except Exception as e:
                            logger.warning(f"Error reading log file: {e}")
                    
                    # Update job status
                    self.video_jobs[job_id] = {
                        "status": "generating",
                        "message": f"Generating video... (progress: {progress}%)",
                        "progress": progress,
                        "video_url": None
                    }
                    
                    # Wait before checking again
                    await asyncio.sleep(2)
                
                # Process completed, get output
                stdout, stderr = await process.communicate()
                
                # Check if process was successful
                if process.returncode != 0:
                    logger.error(f"Error running Cosmos command: {stderr.decode()}")
                    # Try to get more detailed error from log file
                    error_details = "Unknown error"
                    log_file = log_file_path
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, "r") as f:
                                log_content = f.read()
                                # Extract last few lines that might contain errors
                                error_details = "\n".join(log_content.splitlines()[-20:])
                        except Exception as e:
                            logger.warning(f"Error reading log file: {e}")
                    
                    raise RuntimeError(f"Cosmos command failed with code {process.returncode}. Error details: {error_details}")
                
                # Update status to processing
                self.video_jobs[job_id] = {
                    "status": "processing",
                    "message": "Processing video files",
                    "progress": 95,
                    "video_url": None
                }
                
                # Check if video was created
                if not os.path.exists(output_video_path):
                    logger.error(f"Video file not found at {output_video_path}")
                    # Try to find any MP4 files in the output directory
                    mp4_files = []
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith(".mp4"):
                                mp4_files.append(os.path.join(root, file))
                    
                    if mp4_files:
                        # Use the first found MP4 file
                        source_path = mp4_files[0]
                        logger.debug(f"Found MP4 file at {source_path}, copying to {output_video_path}")
                        shutil.copy(source_path, output_video_path)
                    else:
                        # No video found, raise error
                        raise FileNotFoundError(f"No video file generated at {output_video_path} or elsewhere in {output_dir}")
                
                # Create a URL for the video file
                video_url = f"/static/videos/{job_id}/video.mp4"
                logger.debug(f"Video URL: {video_url}")
                
                # Update status to complete
                self.video_jobs[job_id] = {
                    "status": "complete",
                    "message": f"Video generation complete. Video available at: {video_url}",
                    "progress": 100,
                    "video_url": video_url
                }
                
                logger.info(f"Video generation complete for job {job_id}")
            
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error generating video for job {job_id}: {str(e)}")
            logger.debug(f"Error details: {error_details}")
            
            # Create error message based on exception type
            error_message = str(e)
            if "GPU" in error_message:
                error_message = "GPU error - check system configuration"
            elif "out of memory" in error_message.lower():
                error_message = "GPU out of memory - try again later"
            elif "file not found" in error_message.lower():
                error_message = "Failed to generate video file - check logs for details"
            elif "timeout" in error_message.lower():
                error_message = "Command timed out - video generation is taking too long"
            
            # Update status to failed
            self.video_jobs[job_id] = {
                "status": "failed",
                "message": f"Error generating video: {error_message}",
                "progress": 0,
                "video_url": None
            }
            
            # Save error details to file for debugging
            try:
                output_dir = f"static/videos/{job_id}"
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/{job_id}.error", "w") as error_file:
                    error_file.write(f"Error: {str(e)}\n\nStack trace:\n{error_details}")
            except Exception as file_error:
                logger.warning(f"Failed to save error details to file: {file_error}")

    async def _process_video_generation_job(self, prompt: str, media_path: str) -> str:
        """
        Wrapper that uses the existing method to generate a video from a prompt.
        Returns the final video_url on success.
        """
        job_id = str(uuid.uuid4())
        logger.warning(f"(wrapper function) Starting with media path {media_path}")
        await self._process_video_generation(job_id, prompt, media_path)
        
        result = self.video_jobs.get(job_id)
        if not result:
            raise RuntimeError("Job result missing from in-memory store")

        if result.get("status") == "complete":
            return result["video_url"]
        else:
            raise RuntimeError(result.get("message", "Unknown error during video generation"))