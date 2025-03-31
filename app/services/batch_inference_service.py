import os
import uuid
import asyncio
import subprocess
import json
import zipfile
import shutil
from typing import Dict, Optional, List, Any
from fastapi import WebSocket
from pathlib import Path

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger()
settings = get_settings()

# In-memory store for batch job statuses
batch_jobs = {}

class BatchInferenceService:
    """Service for handling batch text-to-world generation on multiple GPUs."""
    
    def __init__(self, num_gpus: int = 8):
        """Initialize the batch inference service with number of available GPUs."""
        self.num_gpus = num_gpus
        self.batch_jobs = {}
    
    async def process_batch(self, prompts: List[str], websocket: Optional[WebSocket] = None) -> str:
        """
        Process a batch of prompts across multiple GPUs.
        
        Args:
            prompts: List of text prompts to process
            websocket: Optional WebSocket for real-time updates
            
        Returns:
            batch_id: Unique identifier for the batch
        """
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Limit prompts to available GPUs
        prompts = prompts[:self.num_gpus]
        
        # Create batch directory
        output_dir = Path("static/videos") / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize batch status
        self.batch_jobs[batch_id] = {
            "total": len(prompts),
            "completed": 0,
            "failed": 0,
            "status": "processing",
            "jobs": []
        }
        
        # Create status entry for each job
        for i in range(len(prompts)):
            job_id = f"job_{i}"
            self.batch_jobs[batch_id]["jobs"].append({
                "job_id": job_id,
                "gpu_id": i,
                "status": "pending",
                "message": "Job is queued and waiting to start",
                "progress": 0,
                "video_url": None,
                "prompt": prompts[i]
            })
        
        # Send initial status if WebSocket provided
        if websocket:
            await websocket.send_json({
                "batch_id": batch_id,
                "status": "processing",
                "message": f"Starting batch processing of {len(prompts)} prompts",
                "jobs": self.batch_jobs[batch_id]["jobs"]
            })
        
        # Start batch processing in background as a fully detached task
        task = asyncio.create_task(self._run_batch(batch_id, prompts, output_dir, websocket))
        
        # Detach the task to ensure it runs independently
        task.add_done_callback(lambda _: None)
        
        return batch_id
    
    async def _run_batch(self, batch_id: str, prompts: List[str], output_dir: Path, websocket: Optional[WebSocket] = None):
        """
        Run batch processing of multiple prompts on different GPUs.
        
        Args:
            batch_id: Unique batch identifier
            prompts: List of text prompts
            output_dir: Directory to save outputs
            websocket: Optional WebSocket for updates
        """
        # Save prompts for reference
        with open(output_dir / "prompts.json", "w") as f:
            json.dump({"prompts": prompts}, f, indent=2)
        
        # Start processes for each GPU
        processes = []
        for i, prompt in enumerate(prompts):
            # Update job status
            self._update_job_status(batch_id, i, "starting", 5)
            if websocket:
                await self._send_status_update(batch_id, websocket)
            
            # Create job directory
            job_dir = output_dir / f"job_{i}"
            job_dir.mkdir(exist_ok=True)
            
            # Prepare command to run on specific GPU
            cmd = [
                "bash", "-c",
                f"cd /workspace/Cosmos && "
                f"CUDA_VISIBLE_DEVICES={i} "
                f"python cosmos1/models/diffusion/inference/text2world.py "
                f"--checkpoint_dir checkpoints "
                f"--diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World "
                f"--prompt \"{prompt}\" "
                f"--disable_prompt_upsampler "
                f"--video_save_name job_{i} "
                f"--video_save_folder {job_dir.absolute()} "
                f"> {job_dir.absolute()}/log.txt 2>&1"
            ]
            
            # Start process
            logger.info(f"Starting job job_{i} on GPU {i} for batch {batch_id}")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            processes.append((i, process))
            
            # Update job status
            self._update_job_status(batch_id, i, "running", 10)
            if websocket:
                await self._send_status_update(batch_id, websocket)
        
        # Monitor processes
        logger.info(f"Monitoring {len(processes)} processes for batch {batch_id}")
        
        while processes:
            for job_idx, process in processes[:]:
                if process.returncode is not None:
                    # Process has completed
                    job_dir = output_dir / f"job_{job_idx}"
                    video_files = list(job_dir.glob("*.mp4"))
                    
                    if video_files and process.returncode == 0:
                        # Success - video generated
                        # Store direct static path for easy frontend embedding
                        video_url = f"/static/videos/{batch_id}/job_{job_idx}/{video_files[0].name}"
                        self._update_job_status(batch_id, job_idx, "complete", 100, video_url)
                        self.batch_jobs[batch_id]["completed"] += 1
                        logger.info(f"Job job_{job_idx} for batch {batch_id} completed successfully")
                    else:
                        # Failure - no video generated
                        self._update_job_status(batch_id, job_idx, "failed", 0)
                        self.batch_jobs[batch_id]["failed"] += 1
                        logger.error(f"Job job_{job_idx} for batch {batch_id} failed")
                    
                    # Remove from active processes
                    processes.remove((job_idx, process))
                else:
                    # Still running - update progress
                    current_progress = self.batch_jobs[batch_id]["jobs"][job_idx]["progress"]
                    if current_progress < 90:  # Cap at 90% until completion
                        new_progress = min(current_progress + 5, 90)
                        self._update_job_status(batch_id, job_idx, "running", new_progress)
            
            # Send status update
            if websocket:
                await self._send_status_update(batch_id, websocket)
            
            # Wait before next check
            await asyncio.sleep(2)  # More frequent updates for better user experience
        
        # All processes completed - update final batch status
        completed = self.batch_jobs[batch_id]["completed"]
        total = self.batch_jobs[batch_id]["total"]
        failed = self.batch_jobs[batch_id]["failed"]
        
        if failed == total:
            self.batch_jobs[batch_id]["status"] = "failed"
            message = "All jobs failed"
        elif failed > 0:
            self.batch_jobs[batch_id]["status"] = "partial"
            message = f"{completed}/{total} jobs completed successfully"
        else:
            self.batch_jobs[batch_id]["status"] = "complete"
            message = "All jobs completed successfully"
        
        logger.info(f"Batch {batch_id} processing completed: {message}")
        
        # Send final status update
        if websocket:
            await self._send_status_update(batch_id, websocket, message)
    
    def _update_job_status(self, batch_id: str, job_idx: int, status: str, progress: int, video_url: str = None):
        """Update the status of a specific job in the batch."""
        if batch_id in self.batch_jobs and 0 <= job_idx < len(self.batch_jobs[batch_id]["jobs"]):
            # Use the same status enum values as single inference (for consistency)
            # "pending", "generating", "processing", "complete", "failed"
            status_map = {
                "pending": "pending",
                "starting": "pending",
                "running": "generating",
                "complete": "complete", 
                "failed": "failed"
            }
            
            mapped_status = status_map.get(status, status)
            
            self.batch_jobs[batch_id]["jobs"][job_idx]["status"] = mapped_status
            self.batch_jobs[batch_id]["jobs"][job_idx]["progress"] = progress
            self.batch_jobs[batch_id]["jobs"][job_idx]["message"] = self._get_status_message(mapped_status)
            
            if video_url:
                self.batch_jobs[batch_id]["jobs"][job_idx]["video_url"] = video_url
    
    def _get_status_message(self, status: str) -> str:
        """Get a human-readable message for a status."""
        status_messages = {
            "pending": "Job is queued and waiting to start",
            "generating": "Generating video from prompt",
            "processing": "Processing generated video",
            "complete": "Video generation complete",
            "failed": "Video generation failed"
        }
        return status_messages.get(status, "Unknown status")
    
    async def _send_status_update(self, batch_id: str, websocket: WebSocket, message: str = None):
        """Send a status update through the WebSocket."""
        if batch_id in self.batch_jobs:
            status_data = {
                "batch_id": batch_id,
                "status": self.batch_jobs[batch_id]["status"],
                "jobs": self.batch_jobs[batch_id]["jobs"],
                "completed": self.batch_jobs[batch_id]["completed"],
                "total": self.batch_jobs[batch_id]["total"]
            }
            
            if message:
                status_data["message"] = message
                
            await websocket.send_json(status_data)
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the current status of a batch."""
        if batch_id not in self.batch_jobs:
            return {
                "batch_id": batch_id,
                "status": "not_found",
                "message": "Batch not found",
                "total": 0,
                "completed": 0,
                "failed": 0,
                "jobs": []
            }
        
        return {
            "batch_id": batch_id,
            "status": self.batch_jobs[batch_id]["status"],
            "jobs": self.batch_jobs[batch_id]["jobs"],
            "completed": self.batch_jobs[batch_id]["completed"],
            "failed": self.batch_jobs[batch_id]["failed"],
            "total": self.batch_jobs[batch_id]["total"],
            "message": f"Completed: {self.batch_jobs[batch_id]['completed']}/{self.batch_jobs[batch_id]['total']}"
        }
        
    async def create_batch_zip(self, batch_id: str) -> Optional[Path]:
        """
        Create a ZIP file containing all completed videos for a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Path to the ZIP file, or None if creation failed
        """
        if batch_id not in self.batch_jobs:
            logger.error(f"Batch {batch_id} not found")
            return None
            
        # Get all completed jobs with videos
        completed_jobs = [
            job for job in self.batch_jobs[batch_id]["jobs"]
            if job["status"] == "complete" and job["video_url"]
        ]
        
        if not completed_jobs:
            logger.error(f"No completed videos found in batch {batch_id}")
            return None
            
        try:
            # Create batch directory if needed
            batch_dir = Path("static/videos") / batch_id
            batch_dir.mkdir(exist_ok=True)
            
            # Create ZIP file
            zip_path = batch_dir / f"batch_{batch_id}_videos.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                for job in completed_jobs:
                    # Extract video path from URL
                    if not job["video_url"]:
                        continue
                        
                    video_url = job["video_url"]
                    video_path = Path(video_url.lstrip('/'))
                    
                    if not video_path.exists():
                        logger.warning(f"Video file not found: {video_path}")
                        continue
                    
                    # Add video to ZIP with job_id and prompt as filename
                    job_id = job["job_id"]
                    safe_prompt = job["prompt"][:30].replace(" ", "_").replace("/", "_")
                    zip_filename = f"{job_id}_{safe_prompt}.mp4"
                    
                    zip_file.write(video_path, zip_filename)
                    
                # Add a README with prompt information
                readme_content = "# Batch Video Generation\n\n"
                readme_content += f"Batch ID: {batch_id}\n"
                readme_content += f"Generated: {len(completed_jobs)} videos\n\n"
                readme_content += "## Videos:\n\n"
                
                for job in completed_jobs:
                    readme_content += f"- {job['job_id']}: \"{job['prompt']}\"\n"
                
                zip_file.writestr("README.txt", readme_content)
            
            logger.info(f"Created ZIP file for batch {batch_id} at {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error creating ZIP for batch {batch_id}: {str(e)}")
            return None