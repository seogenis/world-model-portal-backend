from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import os
import asyncio
import time
from pathlib import Path
from fastapi.responses import FileResponse

from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager
from app.services.video_service import VideoService
from app.services.batch_inference_service import BatchInferenceService
from app.api.schemas import (
    InitializeRequest,
    UpdateRequest,
    PromptResponse,
    EnhancePromptRequest,
    EnhancePromptResponse,
    GenerateVariationsRequest,
    GenerateVariationsResponse,
    PromptHistoryResponse,
    PromptHistoryItem,
    VideoGenerationRequest,
    BatchVideoGenerationRequest,
    VideoStatusResponse,
    BatchVideoStatusResponse,
    BatchJobStatus,
    BatchStatus,
    VideoStatus
)
from app.core.config import get_settings
from app.core.logger import get_logger

router = APIRouter()
settings = get_settings()
logger = get_logger()

# In-memory store for the prompt manager to simulate persistence
# In a production app, this would be a database or Redis
prompt_manager_instance = None
video_service_instance = None
batch_inference_service_instance = None

# Create static directory for videos if it doesn't exist
os.makedirs("static/videos", exist_ok=True)


# Dependency injection
def get_parameter_extractor():
    """Dependency for parameter extractor service."""
    return ParameterExtractor(settings.OPENAI_API_KEY)


def get_prompt_manager(parameter_extractor: ParameterExtractor = Depends(get_parameter_extractor)):
    """Dependency for prompt manager service with session persistence."""
    global prompt_manager_instance
    if prompt_manager_instance is None:
        prompt_manager_instance = PromptManager(parameter_extractor)
    return prompt_manager_instance


def get_video_service():
    """Dependency for video service."""
    global video_service_instance
    if video_service_instance is None:
        video_service_instance = VideoService()
    return video_service_instance


def get_batch_inference_service():
    """Dependency for batch inference service."""
    global batch_inference_service_instance
    if batch_inference_service_instance is None:
        batch_inference_service_instance = BatchInferenceService(num_gpus=8)  # Configure for 8 GPUs
    return batch_inference_service_instance


@router.post("/enhance", response_model=EnhancePromptResponse)
async def enhance_prompt(
    request: EnhancePromptRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Enhance a rough prompt with descriptive details."""
    try:
        enhanced_prompt = await prompt_manager.enhance_prompt(request.rough_prompt)
        return EnhancePromptResponse(
            original_prompt=request.rough_prompt,
            enhanced_prompt=enhanced_prompt
        )
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enhancing prompt: {str(e)}")


@router.post("/initialize", response_model=PromptResponse)
async def initialize_prompt(
    request: InitializeRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Initialize the system with a new prompt."""
    try:
        parameters = await prompt_manager.initialize_from_prompt(request.prompt)
        return PromptResponse(
            parameters=parameters,
            prompt=request.prompt,
            changes=[]
        )
    except Exception as e:
        logger.error(f"Error initializing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing prompt: {str(e)}")


@router.post("/update", response_model=PromptResponse)
async def update_prompt(
    request: UpdateRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Update parameters based on user request."""
    try:
        if not prompt_manager.current_parameters:
            raise HTTPException(
                status_code=400, 
                detail="No prompt initialized. Please initialize a prompt first."
            )
            
        parameters, changes = await prompt_manager.process_update_request(request.user_request)
        new_prompt = await prompt_manager.regenerate_prompt()
        return PromptResponse(
            parameters=parameters,
            prompt=new_prompt,
            changes=changes
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating prompt: {str(e)}")


@router.get("/history", response_model=PromptHistoryResponse)
async def get_history(prompt_manager: PromptManager = Depends(get_prompt_manager)):
    """Get history of all prompts."""
    if not prompt_manager.current_parameters:
        raise HTTPException(
            status_code=400,
            detail="No prompt initialized. Please initialize a prompt first."
        )
    
    prompt_history = prompt_manager.get_prompt_history()
    history_items = []
    
    for item in prompt_history:
        history_items.append(
            PromptHistoryItem(
                prompt=item["prompt"],
                parameters=item["parameters"],
                description=item["description"]
            )
        )
    
    return PromptHistoryResponse(history=history_items)


@router.post("/generate-variations", response_model=GenerateVariationsResponse)
async def generate_variations(
    request: GenerateVariationsRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Generate variations of selected prompts."""
    try:
        if not prompt_manager.get_prompt_history():
            raise HTTPException(
                status_code=400,
                detail="No prompts in history. Please initialize a prompt first."
            )
        
        # Check if all indices are valid
        prompt_history = prompt_manager.get_prompt_history()
        for idx in request.selected_indices:
            if idx < 0 or idx >= len(prompt_history):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid prompt index: {idx}. Valid range is 0-{len(prompt_history)-1}."
                )
        
        variations = await prompt_manager.generate_prompt_variations(
            request.selected_indices, 
            request.total_count
        )
        
        return GenerateVariationsResponse(
            prompts=variations,
            selected_indices=request.selected_indices
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prompt variations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt variations: {str(e)}")


@router.get("/parameters", response_model=Dict[str, Any])
async def get_parameters(prompt_manager: PromptManager = Depends(get_prompt_manager)):
    """Get current parameters."""
    if not prompt_manager.current_parameters:
        raise HTTPException(
            status_code=400,
            detail="No prompt initialized. Please initialize a prompt first."
        )
    return prompt_manager.current_parameters


@router.post("/video/single_inference", response_model=Dict[str, str])
async def generate_single_video(
    request: VideoGenerationRequest,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Generate a video from a text prompt using Nvidia API.
    
    This endpoint initiates the video generation process and returns a job ID.
    The client should connect to the WebSocket endpoint to receive real-time updates.
    """
    try:
        job_id = await video_service.generate_video(request.prompt)
        return {"job_id": job_id, "message": "Video generation started. Connect to WebSocket for updates."}
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting video generation: {str(e)}")


@router.get("/video/status/{job_id}", response_model=VideoStatusResponse)
async def get_video_status(
    job_id: str,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Get the current status of a single video generation job.
    
    This endpoint provides a REST API for polling for status updates.
    """
    try:
        # Check if the job_id is a directory in static/videos
        video_dir = Path("static/videos") / job_id
        
        if video_dir.exists() and video_dir.is_dir():
            # Look for a video file
            video_path = video_dir / "video.mp4"
            if video_path.exists():
                # We found a video, so the job is complete
                return VideoStatusResponse(
                    job_id=job_id,
                    status="complete",
                    message="Video generation complete",
                    progress=100,
                    video_url=f"/api/videos/{job_id}"
                )
            
            # Look for other videos in the directory (check for GPU-specific outputs)
            sub_dirs = list(video_dir.glob("**/"))
            for sub_dir in sub_dirs:
                if sub_dir != video_dir:  # Skip the parent dir
                    video_files = list(sub_dir.glob("*.mp4"))
                    if video_files:
                        logger.info(f"Found video in subdirectory: {video_files[0]}")
                        return VideoStatusResponse(
                            job_id=job_id,
                            status="complete",
                            message="Video generation complete",
                            progress=100,
                            video_url=f"/api/videos/{job_id}"
                        )
                        
            # If we get here, the directory exists but no video yet
            # Check how long we've been waiting
            dir_created_time = video_dir.stat().st_ctime
            current_time = time.time()
            elapsed_time = current_time - dir_created_time
            
            # Estimate progress based on elapsed time (typical video takes 60-120 seconds)
            estimated_progress = min(95, int(elapsed_time / 120 * 100))
            
            return VideoStatusResponse(
                job_id=job_id,
                status="generating",
                message="Video generation in progress...",
                progress=estimated_progress,
                video_url=None
            )
        
        # If we don't find the directory, check batch directories
        batch_dirs = list(Path("static/videos").glob("*"))
        for batch_dir in batch_dirs:
            if not batch_dir.is_dir():
                continue
                
            # Check if it's a batch with job_0, job_1, etc.
            job_dir = batch_dir / f"job_{job_id.split('_')[-1]}" if job_id.startswith("job_") else None
            
            if job_dir and job_dir.exists():
                video_files = list(job_dir.glob("*.mp4"))
                if video_files:
                    return VideoStatusResponse(
                        job_id=job_id,
                        status="complete",
                        message="Video generation complete",
                        progress=100,
                        video_url=f"/api/videos/{job_id}"
                    )
        
        # If we still don't find anything, return a pending status
        # Assume the job is still in early stages
        return VideoStatusResponse(
            job_id=job_id,
            status="pending",
            message="Job is queued and will start soon...",
            progress=10,
            video_url=None
        )
    except Exception as e:
        logger.error(f"Error getting video status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving video status: {str(e)}"
        )


@router.websocket("/ws/video/{job_id}")
async def video_status_websocket(websocket: WebSocket, job_id: str, video_service: VideoService = Depends(get_video_service)):
    """
    WebSocket connection for real-time video generation status updates.
    
    Clients should connect to this endpoint after initiating a video generation job.
    The server will send updates about the job status until it completes or fails.
    """
    await websocket.accept()
    
    try:
        # Initialize a default status that we'll send while waiting for the real status
        default_status = {
            "status": "pending",
            "message": "Waiting for job to start...",
            "progress": 5,
            "video_url": None
        }
        
        # Ensure video_jobs attribute exists in video_service
        if not hasattr(video_service, 'video_jobs'):
            logger.warning(f"video_jobs attribute missing in VideoService for job {job_id}, initializing empty dict")
            video_service.video_jobs = {}
        
        # Send initial status
        if job_id in video_service.video_jobs:
            # If job exists, send its current status
            await websocket.send_json(video_service.video_jobs[job_id])
            
            # If the job is already complete or failed, close the connection
            if video_service.video_jobs[job_id]["status"] in ["complete", "failed"]:
                await websocket.close()
                return
        else:
            # Otherwise send a default "pending" status
            await websocket.send_json(default_status)
        
        # Wait for updates from the video generation process
        max_wait_time = 30 * 60  # 30 minutes max wait time
        wait_interval = 1  # 1 second between checks
        waited_time = 0
        
        while waited_time < max_wait_time:
            # Keep the connection open and check for status updates
            await asyncio.sleep(wait_interval)
            waited_time += wait_interval
            
            # Make sure video_jobs exists (should already be set from earlier check)
            if not hasattr(video_service, 'video_jobs'):
                video_service.video_jobs = {}
                logger.warning(f"video_jobs attribute still missing in VideoService during polling for job {job_id}")
            
            # Check if the job now exists in video_jobs
            if job_id in video_service.video_jobs:
                # Job exists, send current status
                status_data = video_service.video_jobs[job_id]
                await websocket.send_json(status_data)
                
                # Close WebSocket if job is complete or failed
                if status_data["status"] in ["complete", "failed"]:
                    logger.info(f"Job {job_id} status is {status_data['status']}, closing WebSocket")
                    await websocket.close()
                    return
            else:
                # Job doesn't exist yet, send updating status
                default_status["message"] = f"Waiting for job to start... (waited {waited_time}s)"
                default_status["progress"] = min(20, 5 + (waited_time // 10))
                await websocket.send_json(default_status)
                
                # Check if it's a batch job in the current video_service instance
                if hasattr(video_service, 'batch_jobs') and video_service.batch_jobs:
                    for batch_id, batch_data in video_service.batch_jobs.items():
                        for job in batch_data.get('jobs', []):
                            if job.get('job_id') == job_id:
                                # Found as part of a batch, send its status
                                await websocket.send_json(job)
                                
                                # Close if complete or failed
                                if job["status"] in ["complete", "failed"]:
                                    logger.info(f"Batch job {job_id} status is {job['status']}, closing WebSocket")
                                    await websocket.close()
                                    return
                                    
                # Fallback to checking the global video_service_instance as well
                elif video_service_instance and hasattr(video_service_instance, 'batch_jobs'):
                    for batch_id, batch_data in video_service_instance.batch_jobs.items():
                        for job in batch_data.get('jobs', []):
                            if job.get('job_id') == job_id:
                                # Found as part of a batch, send its status
                                await websocket.send_json(job)
                                
                                # Close if complete or failed
                                if job["status"] in ["complete", "failed"]:
                                    logger.info(f"Batch job {job_id} status is {job['status']}, closing WebSocket")
                                    await websocket.close()
                                    return
        
        # If we get here, we've waited too long
        await websocket.send_json({
            "status": "failed",
            "message": "Timed out waiting for job to complete",
            "progress": 0
        })
        await websocket.close()
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for job {job_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            # Try to send an error message before closing
            await websocket.send_json({
                "status": "failed",
                "message": f"Error: {str(e)}",
                "progress": 0
            })
        except:
            pass
        await websocket.close()


@router.post("/video/batch_inference", response_model=Dict[str, str])
async def generate_batch_videos(
    request: BatchVideoGenerationRequest,
    batch_service: BatchInferenceService = Depends(get_batch_inference_service)
):
    """
    Generate multiple videos in parallel using different GPUs.
    
    This endpoint accepts a list of prompts and processes them in parallel
    across different GPUs, returning a batch ID for tracking.
    """
    try:
        if len(request.prompts) > 8:
            logger.warning(f"Too many prompts ({len(request.prompts)}). Limited to 8.")
            
        batch_id = await batch_service.process_batch(request.prompts)
        
        return {
            "batch_id": batch_id,
            "message": f"Batch processing started for {len(request.prompts)} prompts. Connect to WebSocket for updates."
        }
    except Exception as e:
        logger.error(f"Error starting batch video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting batch video generation: {str(e)}")


@router.get("/video/batch_status/{batch_id}")
async def get_batch_status(
    batch_id: str,
    batch_service: BatchInferenceService = Depends(get_batch_inference_service)
):
    """
    Get the current status of a batch generation job.
    
    Returns detailed status for all jobs in the batch.
    """
    return batch_service.get_batch_status(batch_id)


@router.get("/videos/{video_id}")
async def get_video_by_id(video_id: str):
    """
    Unified video retrieval endpoint that works for both single and batch videos.
    
    This endpoint can retrieve:
    - Single videos by their job_id
    - Batch videos by their job_id
    - Raw video files by filename (legacy support)
    
    Returns the video file for download or embedding.
    """
    project_root = Path(__file__).parent.parent.parent
    static_dir = project_root / "static" / "videos"
    
    # Check if it's a direct job directory with video.mp4
    job_dir = static_dir / video_id
    if job_dir.exists() and job_dir.is_dir():
        # First check for video.mp4 in the job directory
        video_path = job_dir / "video.mp4"
        if video_path.exists():
            return FileResponse(
                path=str(video_path),
                media_type="video/mp4",
                filename=f"{video_id}.mp4"
            )
        
        # Check for *.mp4 files in subdirectories (NVIDIA API response)
        for sub_dir in job_dir.glob("*"):
            if sub_dir.is_dir():
                video_files = list(sub_dir.glob("*.mp4"))
                if video_files:
                    return FileResponse(
                        path=str(video_files[0]),
                        media_type="video/mp4",
                        filename=f"{video_id}.mp4"
                    )
    
    # Check if it's a batch job video (format: batch_id/job_id/video.mp4 or batch_id/job_id/job_id.mp4)
    for batch_dir in static_dir.glob("*"):
        if not batch_dir.is_dir():
            continue
        
        # Check for standard batch job directories (job_0, job_1, etc.)
        for job_dir in batch_dir.glob(f"job_*"):
            job_id = job_dir.name
            if job_id == video_id or job_id.startswith(video_id):
                video_files = list(job_dir.glob("*.mp4"))
                if video_files:
                    return FileResponse(
                        path=str(video_files[0]),
                        media_type="video/mp4",
                        filename=f"{video_id}.mp4"
                    )
    
    # Check if it's a direct video file
    video_path = static_dir / f"{video_id}.mp4"
    if video_path.exists():
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )
    
    # Legacy support for root directory videos
    base_video_path = project_root / f"{video_id}.mp4"
    if base_video_path.exists():
        return FileResponse(
            path=str(base_video_path),
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )
    
    # One more fallback - look for any mp4 file related to this job_id anywhere in static/videos
    all_mp4_files = list(static_dir.glob(f"**/{video_id}*.mp4"))
    if all_mp4_files:
        return FileResponse(
            path=str(all_mp4_files[0]), 
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )
        
    # Video not found
    raise HTTPException(
        status_code=404,
        detail=f"Video {video_id} not found"
    )


@router.get("/video/batch_download/{batch_id}")
async def download_batch_videos(
    batch_id: str,
    batch_service: BatchInferenceService = Depends(get_batch_inference_service)
):
    """
    Download all successfully generated videos from a batch as a ZIP file.
    
    This endpoint creates a ZIP file containing all completed videos from the batch
    and returns it for download.
    """
    try:
        # Get batch status to check if videos are available
        status = batch_service.get_batch_status(batch_id)
        
        if status["completed"] == 0:
            raise HTTPException(
                status_code=404,
                detail="No completed videos found in this batch"
            )
        
        # Generate ZIP file of videos
        zip_path = await batch_service.create_batch_zip(batch_id)
        
        if not zip_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to create ZIP file"
            )
        
        # Return the ZIP file
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"batch_{batch_id}_videos.zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch ZIP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating batch ZIP: {str(e)}")


@router.websocket("/ws/batch/{batch_id}")
async def batch_status_websocket(
    websocket: WebSocket,
    batch_id: str,
    batch_service: BatchInferenceService = Depends(get_batch_inference_service)
):
    """
    WebSocket connection for real-time batch generation status updates.
    
    Clients should connect to this endpoint after initiating a batch generation job
    to receive real-time updates about all jobs in the batch.
    """
    await websocket.accept()
    
    try:
        # Send initial status
        status = batch_service.get_batch_status(batch_id)
        await websocket.send_json(status)
        
        # If batch already completed, close connection
        if status["status"] in ["complete", "failed", "not_found"]:
            await websocket.close()
            return
        
        # Otherwise, wait for updates
        while True:
            # Keep connection open for updates from batch service
            await asyncio.sleep(5)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for batch {batch_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        await websocket.close()