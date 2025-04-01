import os
import uuid
import asyncio
import subprocess
import json
import zipfile
import shutil
from typing import Dict, Optional, List, Any
from pathlib import Path

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger()
settings = get_settings()

class BatchInferenceService:
    """Service for handling batch text-to-world generation on multiple GPUs."""
    
    def __init__(self, num_gpus: int = 8):
        """Initialize the batch inference service with number of available GPUs."""
        self.num_gpus = num_gpus
        self.batch_jobs = {}
        
        # Worker pool configuration for background processing
        self.job_queue = asyncio.Queue()
        self.worker_task = None
        
    async def startup(self):
        """Start the worker pool - must be called from an async context"""
        if self.worker_task is None:
            # Start worker pool
            self.worker_task = asyncio.create_task(self._job_worker())
            # Make sure worker doesn't die if there's an exception
            self.worker_task.add_done_callback(self._restart_worker)
            logger.info("Batch inference service worker started")
        
    def _restart_worker(self, future):
        """Callback to restart the worker if it crashes"""
        if future.exception():
            logger.error(f"Batch worker crashed with exception: {future.exception()}")
            # Need to restart in an async context - we'll do this on the next API call
            self.worker_task = None
            logger.info("Batch worker task will be restarted on next API call")
            
    async def _job_worker(self):
        """Worker that processes jobs from the queue"""
        logger.info("Starting batch video generation worker pool")
        while True:
            try:
                # Get next job with timeout
                batch_id, prompts, output_dir, websocket = await asyncio.wait_for(
                    self.job_queue.get(), timeout=1.0
                )
                
                # Start the batch processing
                logger.info(f"Starting batch processing for batch {batch_id}")
                task = asyncio.create_task(
                    self._run_batch(batch_id, prompts, output_dir, websocket)
                )
                
                # Mark task as done in queue
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                # No job available, continue
                pass
            except Exception as e:
                logger.error(f"Error in batch job worker: {e}")
                await asyncio.sleep(1)  # Avoid tight loop in case of persistent errors
    
    async def process_batch(self, prompts: List[str]) -> str:
        """
        Process a batch of prompts across multiple GPUs.
        
        Args:
            prompts: List of text prompts to process
            
        Returns:
            batch_id: Unique identifier for the batch
        """
        # Ensure worker is running
        await self.startup()
        
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Limit prompts to available GPUs
        prompts = prompts[:self.num_gpus]
        
        # Create batch directory
        output_dir = Path("static/videos") / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompts for reference
        with open(output_dir / "prompts.json", "w") as f:
            json.dump({"prompts": prompts}, f, indent=2)
        
        # Initialize batch status
        self.batch_jobs[batch_id] = {
            "total": len(prompts),
            "completed": 0,
            "failed": 0,
            "status": "processing",
            "message": f"Starting batch processing of {len(prompts)} prompts",
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
        
        # WebSocket updates removed
        
        # Add job to queue for processing by worker pool
        await self.job_queue.put((batch_id, prompts, output_dir, None))  # Pass None for websocket parameter
        
        return batch_id
    
    async def _run_batch(self, batch_id: str, prompts: List[str], output_dir: Path, websocket=None):
        """
        Run batch processing of multiple prompts on different GPUs.
        
        Args:
            batch_id: Unique batch identifier
            prompts: List of text prompts
            output_dir: Directory to save outputs
            websocket: Parameter kept for backward compatibility
        """
        # Create jobs list
        jobs = []
        for i, prompt in enumerate(prompts):
            job_id = f"job_{i}"
            gpu_id = i % self.num_gpus  # Assign each prompt to a GPU
            
            # Create job directory
            job_dir = output_dir / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Queue the job for processing
            job = asyncio.create_task(
                self._process_single_job(batch_id, job_id, prompt, job_dir, gpu_id, websocket)
            )
            jobs.append(job)
            
            # Send update via WebSocket
            self._update_job_status(batch_id, job_id, "starting", f"Starting job on GPU {gpu_id}", 5)
            if websocket:
                try:
                    await websocket.send_json(self.get_batch_status(batch_id))
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket update: {e}")
        
        # Wait for all jobs to complete
        try:
            await asyncio.gather(*jobs)
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        
        # Update final batch status
        self._update_batch_status(batch_id)
        
        # Send final update via WebSocket
        # WebSocket updates removed
    
    async def _process_single_job(self, batch_id, job_id, prompt, job_dir, gpu_id, websocket=None):  # websocket param kept for compatibility
        """Process a single job in the batch on the specified GPU."""
        try:
            # Update job status
            self._update_job_status(batch_id, job_id, "starting", f"Starting job on GPU {gpu_id}", 5)
            
            # WebSocket updates removed
            
            # Set environment variable for GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Update job status
            self._update_job_status(batch_id, job_id, "generating", f"Generating video on GPU {gpu_id}", 10)
            if websocket:
                try:
                    await websocket.send_json(self.get_batch_status(batch_id))
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket update: {e}")
            
            # Run text2world command on the local GPU
            # Prepare the prompt
            prompt_file = job_dir / "prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)
            
            # Prepare command for local Cosmos model - using the exact command from the provided code
            cosmos_cmd = [
                "bash", "-c",
                f"cd /workspace/Cosmos && "
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                f"python cosmos1/models/diffusion/inference/text2world.py "
                f"--checkpoint_dir checkpoints "
                f"--diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World "
                f"--prompt \"{prompt}\" "
                f"--disable_prompt_upsampler "
                f"--video_save_name {job_id} "
                f"--video_save_folder {job_dir.absolute()} "
                f"> {job_dir.absolute()}/log.txt 2>&1"
            ]
            
            # Run the cosmos model process
            process = await asyncio.create_subprocess_exec(
                *cosmos_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor progress while process is running
            while process.returncode is None:
                # Check if process has completed
                try:
                    returncode = process.returncode
                    if returncode is not None:
                        break
                except (asyncio.CancelledError, Exception) as e:
                    logger.error(f"Error checking process status: {e}")
                    break
                
                # Read log file to estimate progress if available
                log_file = job_dir / "log.txt"
                current_progress = 10  # Start at 10%
                
                if log_file.exists():
                    try:
                        with open(log_file, "r") as f:
                            log_content = f.read()
                            # Calculate progress based on log content (e.g., "Step 50/100")
                            # This is a simplification - adjust based on actual log format
                            if "Step" in log_content:
                                # Look for patterns like "Step 75/100"
                                import re
                                matches = re.findall(r"Step (\d+)/(\d+)", log_content)
                                if matches:
                                    current, total = map(int, matches[-1])
                                    current_progress = min(90, 10 + int(80 * current / total))
                    except Exception as e:
                        logger.warning(f"Error reading log file: {e}")
                
                # Update job status with current progress
                self._update_job_status(
                    batch_id, job_id, "generating", 
                    f"Generating video on GPU {gpu_id} (Step {current_progress}%)", 
                    current_progress
                )
                
                # Send WebSocket update
                # WebSocket updates removed
                
                # Wait before checking again
                await asyncio.sleep(2)
            
            # Wait for process to complete
            stdout, stderr = await process.communicate()
            
            # Check if process was successful
            if process.returncode != 0:
                logger.error(f"Error generating video for job {job_id}: {stderr.decode()}")
                raise RuntimeError(f"Error generating video: {stderr.decode()}")
            
            # Update status to processing
            self._update_job_status(batch_id, job_id, "processing", "Processing video", 95)
            if websocket:
                try:
                    await websocket.send_json(self.get_batch_status(batch_id))
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket update: {e}")
            
            # Check if video was created - look for any .mp4 files in the job directory
            video_files = list(job_dir.glob("*.mp4"))
            if not video_files:
                raise FileNotFoundError(f"No video files found in {job_dir}")
            
            # Use the first video file found
            video_path = video_files[0]
            
            # Create URL for the video - match the format expected by the frontend
            video_url = f"/static/videos/{batch_id}/{job_id}/{video_path.name}"
            
            # Update job status to complete
            self._update_job_status(batch_id, job_id, "complete", "Video generation complete", 100, video_url)
            
            # Update batch status
            self._update_batch_status(batch_id)
            
            # Send update via WebSocket
            if websocket:
                try:
                    await websocket.send_json(self.get_batch_status(batch_id))
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket update: {e}")
            
            logger.info(f"Completed job {job_id} in batch {batch_id} on GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id} in batch {batch_id}: {str(e)}")
            
            # Update job status to failed
            self._update_job_status(batch_id, job_id, "failed", f"Error: {str(e)}", 0)
            
            # Update batch status
            self._update_batch_status(batch_id)
            
            # Send update via WebSocket
            if websocket:
                try:
                    await websocket.send_json(self.get_batch_status(batch_id))
                except Exception as websocket_error:
                    logger.warning(f"Failed to send WebSocket update: {websocket_error}")
    
    def _update_job_status(self, batch_id, job_id, status, message, progress, video_url=None):
        """Update the status of a job in a batch"""
        if batch_id not in self.batch_jobs:
            logger.warning(f"Batch {batch_id} not found in job store")
            return
            
        batch = self.batch_jobs[batch_id]
        
        # Find the job in the batch
        for job in batch["jobs"]:
            if job["job_id"] == job_id:
                # Update job status
                job["status"] = status
                job["message"] = message
                job["progress"] = progress
                if video_url:
                    job["video_url"] = video_url
                break
    
    def _update_batch_status(self, batch_id):
        """Update the overall batch status based on job statuses"""
        if batch_id not in self.batch_jobs:
            return
            
        batch = self.batch_jobs[batch_id]
        jobs = batch["jobs"]
        
        # Count completed and failed jobs
        completed = sum(1 for job in jobs if job["status"] == "complete")
        failed = sum(1 for job in jobs if job["status"] == "failed")
        
        # Update counts
        batch["completed"] = completed
        batch["failed"] = failed
        
        # Determine overall status
        if completed + failed == len(jobs):
            if failed == len(jobs):
                batch["status"] = "failed"
                batch["message"] = "All jobs failed"
            elif completed == len(jobs):
                batch["status"] = "complete"
                batch["message"] = "All jobs completed successfully"
            else:
                batch["status"] = "partial"
                batch["message"] = f"Completed: {completed}/{len(jobs)}, Failed: {failed}/{len(jobs)}"
        else:
            batch["status"] = "processing"
            batch["message"] = f"Processing: {len(jobs) - completed - failed}/{len(jobs)}"
    
    def get_batch_status(self, batch_id):
        """Get the current status of a batch"""
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
            
        # Update batch status based on jobs
        self._update_batch_status(batch_id)
        
        # Return a copy of the status
        status = dict(self.batch_jobs[batch_id])
        status["batch_id"] = batch_id
        return status
    
    async def create_batch_zip(self, batch_id: str) -> Optional[Path]:
        """
        Create a ZIP file of all videos in a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Path to the ZIP file, or None if creation failed
        """
        if batch_id not in self.batch_jobs:
            logger.warning(f"Batch {batch_id} not found in job store")
            return None
            
        batch_dir = Path("static/videos") / batch_id
        if not batch_dir.exists():
            logger.warning(f"Batch directory not found: {batch_dir}")
            return None
            
        # Create ZIP file
        zip_path = batch_dir / f"batch_{batch_id}_videos.zip"
        
        try:
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                # Get all completed jobs with videos
                completed_jobs = [
                    job for job in self.batch_jobs[batch_id]["jobs"]
                    if job["status"] == "complete" and job["video_url"]
                ]
                
                if not completed_jobs:
                    logger.error(f"No completed videos found in batch {batch_id}")
                    return None
                
                for job in completed_jobs:
                    # Extract video path from URL
                    if not job["video_url"]:
                        continue
                    
                    # Parse the video path from URL format: /static/videos/{batch_id}/{job_id}/{video_name}
                    video_url_parts = job["video_url"].split('/')
                    if len(video_url_parts) >= 5:
                        video_name = video_url_parts[-1]
                        video_path = batch_dir / job["job_id"] / video_name
                    else:
                        logger.warning(f"Invalid video URL format: {job['video_url']}")
                        continue
                    
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