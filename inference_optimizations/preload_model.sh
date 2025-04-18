cd /workspace/Cosmos && \
  export OMP_NUM_THREADS=16 && \
  NVTE_FUSED_ATTN=0 \
  torchrun --nproc_per_node=8 \
  cosmos1/models/diffusion/nemo/inference/preloaded_inference.py \
  --preload_text2world \
  --preload_video2world \
  --num_devices 8 \
  --cp_size 8 \
  --job_dir /workspace/world-model-portal-backend/static/jobs \
  --checkpoints_root /workspace/checkpoints/hub \
