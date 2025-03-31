import os
import uuid
import asyncio
import tempfile
import zipfile
import shutil
import requests
import time
import json
from typing import Dict, Optional, Tuple, Any
from fastapi import WebSocket

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger()
settings = get_settings()

# In-memory store for job statuses (moved inside the class)

class VideoService:
    """Service for handling video generation requests using Nvidia's API."""
    
    def __init__(self):
        """Initialize the video service with API credentials."""
        settings = get_settings()
        self.nvidia_api_key = settings.NVIDIA_API_KEY
        self.invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-predict1-7b"
        self.fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
        self.headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json",
        }
        # In-memory store for job statuses
        self.video_jobs: Dict[str, Dict[str, Any]] = {}
        
    async def generate_video(self, prompt: str, websocket: Optional[WebSocket] = None) -> str:
        """
        Generate a video from a prompt using Nvidia API.
        
        Args:
            prompt: The text-to-video prompt
            websocket: Optional WebSocket connection for real-time updates
            
        Returns:
            job_id: Unique identifier for tracking the video generation
        """
        job_id = str(uuid.uuid4())
        
        # Initialize job in memory store immediately
        self.video_jobs[job_id] = {
            "status": "pending",
            "message": "Job queued and will start soon",
            "progress": 0,
            "video_url": None
        }
        
        # Start video generation in a fully detached background task
        # This ensures it won't block the main application thread
        task = asyncio.create_task(self._process_video_generation(job_id, prompt, websocket))
        
        # Detach the task to ensure it runs independently
        task.add_done_callback(lambda _: None)
        
        # Return job_id immediately without waiting for video generation
        return job_id
        
    async def _process_video_generation(self, job_id: str, prompt: str, websocket: Optional[WebSocket] = None) -> None:
        """
        Process the video generation job.
        
        Args:
            job_id: The job ID
            prompt: The text-to-video prompt
            websocket: Optional WebSocket connection for real-time updates
        """
        try:
            # Send initial status update
            if websocket:
                await websocket.send_json({
                    "status": "pending",
                    "message": "Job queued",
                    "progress": 0
                })
            
            # Create the payload for Nvidia API
            payload = {
                "inputs": [
                    {
                        "name": "command",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [
                            f"text2world --prompt=\"{prompt}\""
                        ]
                    }
                ],
                "outputs": [
                    {
                        "name": "status",
                        "datatype": "BYTES",
                        "shape": [1]
                    }
                ]
            }
            
            # Update status to generating
            if websocket:
                await websocket.send_json({
                    "status": "generating",
                    "message": "Generating video from prompt",
                    "progress": 10
                })
                
            # Create a session for connection reuse
            session = requests.Session()
            
            # Send the initial request
            logger.info(f"Sending request to Nvidia API for job {job_id}")
            response = session.post(self.invoke_url, headers=self.headers, json=payload)
            
            # Poll until the job is complete
            progress = 10
            while response.status_code == 202:
                # Update progress based on time spent (simple approach)
                progress += 5
                progress = min(progress, 90)  # Cap at 90% until we have the result
                
                if websocket:
                    await websocket.send_json({
                        "status": "generating",
                        "message": "Generating video...",
                        "progress": progress
                    })
                
                # Get the request ID and fetch the status
                request_id = response.headers.get("NVCF-REQID")
                fetch_url = self.fetch_url_format + request_id
                
                logger.info(f"Checking status for job {job_id}, request_id {request_id}")
                response = session.get(fetch_url, headers=self.headers)
                
                # Wait a bit before checking again
                await asyncio.sleep(2)
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Update status to processing
            if websocket:
                await websocket.send_json({
                    "status": "processing",
                    "message": "Processing video files",
                    "progress": 95
                })
            
            # Create directory for storing the video
            output_dir = f"static/videos/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Check if response content is a zip file
                content_type = response.headers.get('Content-Type', '')
                if 'zip' in content_type.lower() or response.content[:4] == b'PK\x03\x04':
                    # Save the ZIP file
                    zip_path = f"{output_dir}/result.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract the video file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    
                    # Look for video.mp4 in the extracted files
                    video_path = f"{output_dir}/video.mp4"
                    
                    # If it doesn't exist, look for any .mp4 file
                    if not os.path.exists(video_path):
                        mp4_files = []
                        for root, _, files in os.walk(output_dir):
                            for file in files:
                                if file.endswith(".mp4"):
                                    mp4_files.append(os.path.join(root, file))
                        
                        if mp4_files:
                            # Use the first found mp4 file
                            shutil.copy(mp4_files[0], video_path)
                        else:
                            raise FileNotFoundError("No mp4 files found in extracted zip")
                else:
                    # Not a zip file, check if it's directly an mp4
                    content_start = response.content[:16]
                    is_mp4 = False
                    
                    # Check for mp4 file signature (ftyp...)
                    if len(content_start) > 8 and content_start[4:8] == b'ftyp':
                        is_mp4 = True
                    
                    if is_mp4 or 'video/mp4' in content_type.lower():
                        # Direct mp4 file, save it
                        video_path = f"{output_dir}/video.mp4"
                        with open(video_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        # Unknown format, save response for inspection
                        error_file = f"{output_dir}/response.bin"
                        with open(error_file, 'wb') as f:
                            f.write(response.content)
                        
                        # Save headers for debugging
                        with open(f"{output_dir}/headers.json", 'w') as f:
                            json.dump(dict(response.headers), f, indent=2)
                            
                        raise ValueError(f"Response is not a zip or mp4 file. Content-Type: {content_type}")
            except zipfile.BadZipFile:
                # Not a valid zip file, could be direct mp4 or error response
                # Try to save as mp4
                video_path = f"{output_dir}/video.mp4"
                with open(video_path, 'wb') as f:
                    f.write(response.content)
                
                # Check if it's a valid mp4
                if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
                    # Too small to be a real video, probably error response
                    error_file = f"{output_dir}/response.txt"
                    with open(error_file, 'wb') as f:
                        f.write(response.content)
                    
                    raise ValueError(f"Response appears to be error data, not a valid video")
            
            # Create a URL for the video file
            video_url = f"/static/videos/{job_id}/video.mp4"
            
            # Update status to complete
            if websocket:
                await websocket.send_json({
                    "status": "complete",
                    "message": "Video generation complete",
                    "progress": 100,
                    "video_url": video_url
                })
            
            # Store the result in the in-memory store
            self.video_jobs[job_id] = {
                "status": "complete",
                "message": "Video generation complete",
                "progress": 100,
                "video_url": video_url
            }
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            
            # Update status to failed
            if websocket:
                await websocket.send_json({
                    "status": "failed",
                    "message": f"Error generating video: {str(e)}",
                    "progress": 0
                })
            
            # Store the error in the in-memory store
            self.video_jobs[job_id] = {
                "status": "failed",
                "message": f"Error generating video: {str(e)}",
                "progress": 0
            }