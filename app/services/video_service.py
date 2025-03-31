import os
import uuid
import asyncio
import tempfile
import zipfile
import shutil
import requests
import time
from typing import Dict, Optional, Tuple
from fastapi import WebSocket
import json

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger()
settings = get_settings()

# In-memory store for job statuses
video_jobs = {}

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
        
        # Start video generation in background task
        asyncio.create_task(self._process_video_generation(job_id, prompt, websocket))
        
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
            os.makedirs(f"static/videos/{job_id}", exist_ok=True)
            
            # Save the ZIP file
            zip_path = f"static/videos/{job_id}/result.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the video file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(f"static/videos/{job_id}")
            
            # Check if video.mp4 exists in the extracted files
            video_path = f"static/videos/{job_id}/video.mp4"
            if not os.path.exists(video_path):
                # Look for any .mp4 file in the extracted directory
                for root, _, files in os.walk(f"static/videos/{job_id}"):
                    for file in files:
                        if file.endswith(".mp4"):
                            # Move or copy the file to the expected location
                            shutil.copy(os.path.join(root, file), video_path)
                            break
            
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
            video_jobs[job_id] = {
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
            video_jobs[job_id] = {
                "status": "failed",
                "message": f"Error generating video: {str(e)}",
                "progress": 0
            }