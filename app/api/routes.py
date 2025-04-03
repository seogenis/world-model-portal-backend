from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
import os
import asyncio
import time
import uuid
from pathlib import Path
from fastapi.responses import FileResponse

from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager
from app.services.session_manager import SessionManagerService
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
    VideoStatus,
    SessionRequest,
    SessionIdRequest
)
from app.core.config import get_settings
from app.core.logger import get_logger

from celery.result import AsyncResult
from app.celery_worker import celery_app
from app.celery_worker import run_video_generation
import logging

router = APIRouter()
settings = get_settings()
logger = get_logger()

# Create static directory for videos if it doesn't exist
os.makedirs("static/videos", exist_ok=True)


# Dependency injection
def get_parameter_extractor():
    """Dependency for parameter extractor service."""
    return ParameterExtractor(settings.OPENAI_API_KEY)


def get_session_manager(parameter_extractor: ParameterExtractor = Depends(get_parameter_extractor)):
    """Dependency for session manager service."""
    return SessionManagerService(parameter_extractor)


def get_video_service():
    """Dependency for video service."""
    return VideoService()


def get_batch_inference_service():
    """Dependency for batch inference service."""
    return BatchInferenceService(num_gpus=8)  # Configure for 8 GPUs


async def get_prompt_manager_for_session(
    session_id: Optional[str] = None,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """
    Get or create a PromptManager for a session.
    
    If session_id is not provided, a new session will be created.
    """
    if not session_id:
        # Create a new session
        session_id = await session_manager.create_session()
        logger.info(f"Created new session: {session_id}")
    
    # Get the prompt manager for this session
    prompt_manager = await session_manager.get_prompt_manager(session_id)
    
    return prompt_manager, session_id


@router.post("/enhance", response_model=EnhancePromptResponse)
async def enhance_prompt(
    request: EnhancePromptRequest,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Enhance a rough prompt with descriptive details."""
    try:
        # Get or create a session
        session_id = request.session_id
        if not session_id:
            session_id = await session_manager.create_session()
            logger.info(f"Created new session for enhance: {session_id}")
        
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        # Enhance the prompt
        enhanced_prompt = await prompt_manager.enhance_prompt(request.rough_prompt)
        
        # Save the updated prompt manager
        await session_manager._save_prompt_manager(session_id, prompt_manager)
        
        return EnhancePromptResponse(
            original_prompt=request.rough_prompt,
            enhanced_prompt=enhanced_prompt,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enhancing prompt: {str(e)}")


@router.post("/initialize", response_model=PromptResponse)
async def initialize_prompt(
    request: InitializeRequest,
    session_manager: SessionManagerService = Depends(get_session_manager),
    client_host: str = Query("", alias="X-Forwarded-For", include_in_schema=False)
):
    """Initialize the system with a new prompt."""
    try:
        # Get or create a session
        session_id = request.session_id
        if not session_id:
            # Get client IP for rate limiting (using X-Forwarded-For if available)
            client_ip = client_host.split(",")[0].strip() if client_host else "unknown"
            session_id = await session_manager.create_session(ip_address=client_ip)
            logger.info(f"Created new session for initialize: {session_id} from IP {client_ip}")
        
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        # Initialize from prompt
        parameters = await prompt_manager.initialize_from_prompt(request.prompt)
        
        # Save the updated prompt manager
        await session_manager._save_prompt_manager(session_id, prompt_manager)
        
        return PromptResponse(
            parameters=parameters,
            prompt=request.prompt,
            changes=[],
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error initializing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing prompt: {str(e)}")


@router.post("/update", response_model=PromptResponse)
async def update_prompt(
    request: UpdateRequest,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Update parameters based on user request."""
    try:
        # Get the session ID from the request
        session_id = request.session_id
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="No session ID provided. Please initialize a prompt first."
            )
        
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        if not prompt_manager.current_parameters:
            raise HTTPException(
                status_code=400, 
                detail=f"No prompt initialized for session {session_id}. Please initialize a prompt first or check that you're using the correct session ID."
            )
            
        # Process the update request
        parameters, changes = await prompt_manager.process_update_request(request.user_request)
        new_prompt = await prompt_manager.regenerate_prompt()
        
        # Save the updated prompt manager
        await session_manager._save_prompt_manager(session_id, prompt_manager)
        
        return PromptResponse(
            parameters=parameters,
            prompt=new_prompt,
            changes=changes,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating prompt for session {request.session_id}: {str(e)}")


@router.post("/history", response_model=PromptHistoryResponse)
async def get_history(
    request: SessionIdRequest,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Get history of all prompts."""
    try:
        # Get session ID from request body
        session_id = request.session_id
            
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        if not prompt_manager.current_parameters:
            raise HTTPException(
                status_code=400,
                detail=f"No prompt initialized for session {session_id}. Please initialize a prompt first or check that you're using the correct session ID."
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
        
        return PromptHistoryResponse(
            history=history_items,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt history for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prompt history for session {request.session_id}: {str(e)}")


@router.post("/generate-variations", response_model=GenerateVariationsResponse)
async def generate_variations(
    request: GenerateVariationsRequest,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Generate variations of selected prompts."""
    try:
        # Get the session ID from the request
        session_id = request.session_id
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="No session ID provided. Please initialize a prompt first."
            )
        
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        if not prompt_manager.get_prompt_history():
            raise HTTPException(
                status_code=400,
                detail=f"No prompts in history for session {session_id}. Please initialize a prompt first or check that you're using the correct session ID."
            )
        
        # Check if all indices are valid
        prompt_history = prompt_manager.get_prompt_history()
        for idx in request.selected_indices:
            if idx < 0 or idx >= len(prompt_history):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid prompt index: {idx}. Valid range is 0-{len(prompt_history)-1}."
                )
        
        # Generate variations - this now uses Pydantic validation internally
        variations = await prompt_manager.generate_prompt_variations(
            request.selected_indices, 
            request.total_count
        )
        
        # Save the updated prompt manager
        await session_manager._save_prompt_manager(session_id, prompt_manager)
        
        # Double-check we have exactly the requested number of prompts
        if len(variations) != request.total_count:
            logger.warning(f"Expected {request.total_count} variations but got {len(variations)}. Adjusting...")
            
            # If we have too few, duplicate the last ones
            while len(variations) < request.total_count:
                idx = len(variations) % len(variations or [1])
                source = variations[idx] if variations else "A detailed scene with interesting visuals"
                variations.append(f"{source} (api fallback)")
            
            # If we have too many, trim
            variations = variations[:request.total_count]
        
        logger.info(f"API returning exactly {len(variations)} variations")
        
        return GenerateVariationsResponse(
            prompts=variations,
            selected_indices=request.selected_indices,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prompt variations for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt variations for session {request.session_id}: {str(e)}")


@router.post("/parameters")
async def get_parameters(
    request: SessionIdRequest,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Get current parameters."""
    try:
        # Get session ID from request body
        session_id = request.session_id
            
        # Get the prompt manager for this session
        prompt_manager = await session_manager.get_prompt_manager(session_id)
        
        if not prompt_manager.current_parameters:
            raise HTTPException(
                status_code=400,
                detail=f"No prompt initialized for session {session_id}. Please initialize a prompt first or check that you're using the correct session ID."
            )
        
        # Return the current parameters along with the session ID
        return {
            "parameters": prompt_manager.current_parameters,
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parameters for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting parameters for session {request.session_id}: {str(e)}")


@router.get("/sessions")
async def list_sessions(
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """List all active sessions with their last access time."""
    try:
        sessions = await session_manager.list_active_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManagerService = Depends(get_session_manager)
):
    """Delete a session."""
    try:
        success = await session_manager.delete_session(session_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

'''
@router.post("/video/single_inference", response_model=Dict[str, str])
async def generate_single_video(
    request: VideoGenerationRequest,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Generate a video from a text prompt using Nvidia API.
    
    This endpoint initiates the video generation process and returns a job ID.
    The client can check status using the status endpoint.
    """
    try:
        job_id = await video_service.generate_video(request.prompt)
        return {"job_id": job_id, "message": "Video generation started. Check status endpoint for updates."}
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting video generation: {str(e)}")
'''

@router.post("/video/single_inference", response_model=Dict[str, str])
async def generate_single_video(
    request: VideoGenerationRequest,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Generate a video from a text prompt using Nvidia API.
    
    This endpoint initiates the video generation process and returns a job ID.
    The client can check status using the status endpoint.
    """
    try:
        # Submit to Celery (async in background, not using video_service directly)
        task = run_video_generation.delay(request.prompt)

        return {
            "job_id": task.id,
            "message": "Video generation started. Check status endpoint for updates."
        }
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting video generation: {str(e)}")

@router.get("/video/status/{job_id}", response_model=VideoStatusResponse)
async def get_video_status(
    job_id: str,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Get the current status of a video generation task by job ID.
    This queries the Celery task backend (Redis) for task state and result.
    """
    try:
        task = AsyncResult(job_id, app=celery_app)

        if task.state == "PENDING":
            return VideoStatusResponse(
                job_id=job_id,
                status="pending",
                message="Job is queued and will start soon.",
                progress=0,
                video_url=None
            )

        elif task.state == "STARTED":
            return VideoStatusResponse(
                job_id=job_id,
                status="generating",
                message="Video generation in progress...",
                progress=50,
                video_url=None
            )

        elif task.state == "SUCCESS":
            return VideoStatusResponse(
                job_id=job_id,
                status="complete",
                message="Video generation complete.",
                progress=100,
                video_url=task.result  # expected to be a URL
            )

        elif task.state == "FAILURE":
            return VideoStatusResponse(
                job_id=job_id,
                status="failed",
                message=f"Video generation failed: {str(task.result)}",
                progress=0,
                video_url=None
            )

        else:
            return VideoStatusResponse(
                job_id=job_id,
                status=task.state.lower(),
                message="Job is in an unknown state.",
                progress=0,
                video_url=None
            )

    except Exception as e:
        logger.error(f"Error getting video status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving video status: {str(e)}"
        )

'''
@router.get("/video/status/{job_id}", response_model=VideoStatusResponse)
async def get_video_status(
    job_id: str,
    video_service: VideoService = Depends(get_video_service)
):
    """
    Get the current status of a single video generation job.
    
    This endpoint provides a REST API for polling for status updates.
    The API checks static/videos/{job_id} to determine if the job is complete,
    and provides the static file path in the response.
    """
    try:
        # First, check if the video file exists directly in the expected static path
        # This is the primary check for completed jobs
        static_path = f"/static/videos/{job_id}/video.mp4"
        actual_file_path = Path(static_path.lstrip('/'))  # Remove leading slash for filesystem path
        
        if actual_file_path.exists():
            # Video file exists at the expected location, job is complete
            logger.info(f"Status request for job {job_id}: status=complete")
            return VideoStatusResponse(
                job_id=job_id,
                status="complete",
                message=f"Video generation complete. Video available at: {static_path}",
                progress=100,
                video_url=static_path
            )
        
        # Next, check if the job directory exists, might be in progress
        video_dir = Path("static/videos") / job_id
        
        if video_dir.exists() and video_dir.is_dir():
            # Directory exists, check for any other MP4 files
            video_files = list(video_dir.glob("**/*.mp4"))
            
            if video_files:
                # Found a video file in the job directory or subdirectories
                relative_path = video_files[0].relative_to(Path("static"))
                static_path = f"/static/{relative_path}"
                logger.info(f"Found video in directory: {static_path}")
                
                return VideoStatusResponse(
                    job_id=job_id,
                    status="complete",
                    message=f"Video generation complete. Video available at: {static_path}",
                    progress=100,
                    video_url=static_path
                )
            
            # Directory exists but no video yet - job is in progress
            # Check how long we've been waiting
            dir_created_time = video_dir.stat().st_ctime
            current_time = time.time()
            elapsed_time = current_time - dir_created_time
            
            # Estimate progress based on elapsed time (typical video takes 60-120 seconds)
            estimated_progress = min(95, int(elapsed_time / 120 * 100))
            
            logger.info(f"Status request for job {job_id}: status=generating (progress: {estimated_progress}%)")
            
            return VideoStatusResponse(
                job_id=job_id,
                status="generating",
                message=f"Video generation in progress... Expected path when complete: {static_path}",
                progress=estimated_progress,
                video_url=None
            )
        
        # If we don't find the directory, check if this is a batch ID
        # (Sometimes users confuse batch_id with job_id)
        batch_dir = Path("static/videos") / job_id
        if batch_dir.exists() and batch_dir.is_dir():
            # This is likely a batch ID, not a job ID
            batch_job_dirs = [d for d in batch_dir.glob("job_*") if d.is_dir()]
            
            if batch_job_dirs:
                # Treat this as a batch, but return info for first job
                logger.info(f"ID {job_id} appears to be a batch ID with {len(batch_job_dirs)} jobs, not a job ID")
                
                # Find a completed job
                for job_dir in batch_job_dirs:
                    video_files = list(job_dir.glob("**/*.mp4"))
                    if video_files:
                        # Use the first completed job we find
                        actual_job_id = job_dir.name
                        relative_path = video_files[0].relative_to(Path("static"))
                        static_path = f"/static/{relative_path}"
                        
                        logger.info(f"Found completed job {actual_job_id} in batch {job_id}")
                        
                        return VideoStatusResponse(
                            job_id=actual_job_id,  # Return the actual job ID
                            status="complete",
                            message=f"Video generation complete. This is job {actual_job_id} from batch {job_id}. Video available at: {static_path}",
                            progress=100,
                            video_url=static_path
                        )
                
                # If no completed job found, return info about the batch
                return VideoStatusResponse(
                    job_id=f"job_0",  # Default to first job
                    status="processing",
                    message=f"This is a batch ID with {len(batch_job_dirs)} jobs, not a single job ID. Use batch_status endpoint instead.",
                    progress=50,
                    video_url=None
                )
        
        # Check other batch directories for this job
        batch_dirs = list(Path("static/videos").glob("*"))
        for batch_dir in batch_dirs:
            if not batch_dir.is_dir() or batch_dir.name == job_id:  # Skip if it's the directory we already checked
                continue
                
            # Check both naming patterns: job_id or job_{index}
            possible_job_dirs = [
                batch_dir / job_id,  # Direct job ID
                batch_dir / f"job_{job_id.split('_')[-1]}" if job_id.startswith("job_") else None  # job_X pattern
            ]
            
            for job_dir in possible_job_dirs:
                if job_dir and job_dir.exists():
                    video_files = list(job_dir.glob("**/*.mp4"))
                    if video_files:
                        # Found video in batch job directory
                        relative_path = video_files[0].relative_to(Path("static"))
                        static_path = f"/static/{relative_path}"
                        
                        batch_id = batch_dir.name
                        logger.info(f"Found job {job_id} in batch {batch_id}")
                        
                        return VideoStatusResponse(
                            job_id=job_id,
                            status="complete",
                            message=f"Video generation complete. Part of batch {batch_id}. Video available at: {static_path}",
                            progress=100,
                            video_url=static_path
                        )
        
        # Check legacy format (flat file in videos directory)
        legacy_path = Path(f"static/videos/{job_id}.mp4")
        if legacy_path.exists():
            static_path = f"/static/videos/{job_id}.mp4"
            return VideoStatusResponse(
                job_id=job_id,
                status="complete",
                message=f"Video generation complete. Video available at: {static_path}",
                progress=100,
                video_url=static_path
            )
        
        # If we still don't find anything, check if it's in the video service memory
        if hasattr(video_service, 'video_jobs') and job_id in video_service.video_jobs:
            # Return status from video_service if available
            status_data = video_service.video_jobs[job_id]
            
            # Update expected path in messages
            if status_data["status"] == "complete":
                status_data["message"] = f"Video generation complete. Video available at: /static/videos/{job_id}/video.mp4"
            
            logger.info(f"Status request for job {job_id}: status={status_data['status']} (progress: {status_data['progress']}%)")
                
            return VideoStatusResponse(
                job_id=job_id,
                status=status_data["status"],
                message=status_data["message"],
                progress=status_data["progress"],
                video_url=status_data.get("video_url")
            )
        
        # If we still don't find anything, return a pending status
        logger.info(f"Status request for job {job_id}: status=pending (job not found in system)")
        
        return VideoStatusResponse(
            job_id=job_id,
            status="pending",
            message=f"Job is queued and will start soon... Expected path when complete: /static/videos/{job_id}/video.mp4",
            progress=10,
            video_url=None
        )
    except Exception as e:
        logger.error(f"Error getting video status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving video status: {str(e)}"
        )
'''



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
            "message": f"Batch processing started for {len(request.prompts)} prompts. Check batch_status endpoint for updates."
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
    Checks static/videos/{batch_id} for completed videos and updates
    status accordingly with actual file paths, even if the batch is not found
    in the service's memory.
    """
    # First get status from the batch service
    batch_status = batch_service.get_batch_status(batch_id)
    
    # Check if batch directory exists on disk
    batch_dir = Path("static/videos") / batch_id
    if batch_dir.exists() and batch_dir.is_dir():
        logger.info(f"Batch status request for batch {batch_id}: initial_status={batch_status.get('status')}")
        
        # If status is "not_found" but directory exists, create a new status object
        if batch_status.get("status") == "not_found":
            logger.info(f"Batch {batch_id} not found in memory but exists on disk. Scanning directory...")
            
            # Scan directory for job folders
            job_dirs = [d for d in batch_dir.glob("job_*") if d.is_dir()]
            
            if job_dirs:
                # Create new batch status based on filesystem
                batch_status = {
                    "batch_id": batch_id,
                    "status": "unknown",  # Will update this later
                    "message": f"Reconstructed batch from filesystem: {len(job_dirs)} jobs found",
                    "total": len(job_dirs),
                    "completed": 0,
                    "failed": 0,
                    "jobs": []
                }
                
                # Process each job directory
                for job_dir in job_dirs:
                    job_id = job_dir.name
                    
                    # Check for video files in this job's directory
                    video_files = list(job_dir.glob("**/*.mp4"))
                    if video_files:
                        # Found a video file for this job
                        video_file = video_files[0]
                        relative_path = video_file.relative_to(Path("static"))
                        static_path = f"/static/{relative_path}"
                        
                        # Add job to status
                        batch_status["jobs"].append({
                            "job_id": job_id,
                            "status": "complete",
                            "message": f"Video generation complete. Video available at: {static_path}",
                            "progress": 100,
                            "video_url": static_path
                        })
                        
                        batch_status["completed"] += 1
                        logger.info(f"Found completed video for job {job_id} at {static_path}")
                    else:
                        # Job directory exists but no video
                        # Check for error files
                        error_files = list(job_dir.glob("*.error"))
                        if error_files:
                            # Job failed
                            batch_status["jobs"].append({
                                "job_id": job_id,
                                "status": "failed",
                                "message": "Video generation failed",
                                "progress": 0,
                                "video_url": None
                            })
                            batch_status["failed"] += 1
                        else:
                            # Job in progress
                            batch_status["jobs"].append({
                                "job_id": job_id,
                                "status": "processing",
                                "message": "Video generation in progress...",
                                "progress": 50,  # Estimate
                                "video_url": None
                            })
                
                # Update the overall batch status
                completed = batch_status["completed"]
                failed = batch_status["failed"]
                total = batch_status["total"]
                
                if total == 0:
                    batch_status["status"] = "unknown"
                    batch_status["message"] = "Batch exists but no jobs found"
                elif failed == total:
                    batch_status["status"] = "failed"
                    batch_status["message"] = "All jobs failed"
                elif completed == total:
                    batch_status["status"] = "complete"
                    batch_status["message"] = "All jobs completed successfully"
                elif completed > 0 or failed > 0:
                    batch_status["status"] = "partial"
                    batch_status["message"] = f"Completed: {completed}/{total}, Failed: {failed}/{total}"
                else:
                    batch_status["status"] = "processing"
                    batch_status["message"] = "Batch processing in progress"
                
                logger.info(f"Batch status request for batch {batch_id}: status={batch_status['status']} (completed: {completed}/{total}, failed: {failed}/{total})")
            else:
                # Batch directory exists but no job folders found
                logger.warning(f"Batch directory {batch_dir} exists but no job folders found")
                batch_status = {
                    "batch_id": batch_id,
                    "status": "unknown",
                    "message": f"Batch directory exists but no job folders found",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "jobs": []
                }
        else:
            # Batch exists in memory, update existing status with filesystem info
            # Go through each job in the status
            for job in batch_status.get("jobs", []):
                job_id = job.get("job_id")
                # Only check jobs that aren't already marked as complete
                if job_id and job.get("status") != "complete":
                    # Check for video files in this job's directory
                    job_dir = batch_dir / job_id
                    if job_dir.exists() and job_dir.is_dir():
                        video_files = list(job_dir.glob("**/*.mp4"))
                        if video_files:
                            # Job is complete, update the status
                            relative_path = video_files[0].relative_to(Path("static"))
                            static_path = f"/static/{relative_path}"
                            
                            job["status"] = "complete"
                            job["message"] = f"Video generation complete. Video available at: {static_path}"
                            job["progress"] = 100
                            job["video_url"] = static_path
                            
                            logger.info(f"Found completed video for job {job_id} at {static_path}")
                
            # For all jobs marked as complete, ensure they have the right video_url path
            for job in batch_status.get("jobs", []):
                if job.get("status") == "complete" and not job.get("video_url", "").startswith("/static/"):
                    job_id = job.get("job_id")
                    # Set the expected static path
                    job["video_url"] = f"/static/videos/{batch_id}/{job_id}/video.mp4"
                    job["message"] = f"Video generation complete. Video available at: {job['video_url']}"
                    
            # Update the overall batch status based on job statuses
            jobs = batch_status.get("jobs", [])
            if jobs:
                completed = sum(1 for job in jobs if job.get("status") == "complete")
                failed = sum(1 for job in jobs if job.get("status") == "failed")
                
                batch_status["completed"] = completed
                batch_status["failed"] = failed
                
                # Update overall batch status
                if failed == len(jobs):
                    batch_status["status"] = "failed"
                    batch_status["message"] = "All jobs failed"
                elif completed == len(jobs):
                    batch_status["status"] = "complete"
                    batch_status["message"] = "All jobs completed successfully"
                elif completed > 0 or failed > 0:
                    batch_status["status"] = "partial"
                    batch_status["message"] = f"Completed: {completed}/{len(jobs)}, Failed: {failed}/{len(jobs)}"
                    
    elif batch_status.get("status") == "not_found":
        # Check for legacy batch format where videos are directly in static/videos with batch ID prefixes
        # This handles cases where batch_id might be followed by job index in the filename
        potential_batch_videos = list(Path("static/videos").glob(f"{batch_id}*.mp4"))
        if potential_batch_videos:
            logger.info(f"Found {len(potential_batch_videos)} videos with batch ID prefix: {batch_id}")
            
            # Create batch status from found videos
            jobs = []
            for video_file in potential_batch_videos:
                # Extract job ID from filename (after batch ID)
                file_name = video_file.stem
                job_id = file_name.replace(batch_id, "").strip("_-")
                if not job_id:
                    job_id = f"job_{len(jobs)}"  # Create sequential job ID if none found
                
                relative_path = video_file.relative_to(Path("static"))
                static_path = f"/static/{relative_path}"
                
                jobs.append({
                    "job_id": job_id,
                    "status": "complete",
                    "message": f"Video generation complete. Video available at: {static_path}",
                    "progress": 100,
                    "video_url": static_path
                })
            
            batch_status = {
                "batch_id": batch_id,
                "status": "complete",
                "message": f"Found {len(jobs)} completed videos for this batch",
                "total": len(jobs),
                "completed": len(jobs),
                "failed": 0,
                "jobs": jobs
            }
            
    # Log final status before returning
    logger.info(f"Batch status request for batch {batch_id}: final_status={batch_status.get('status')} (completed: {batch_status.get('completed', 0)}/{batch_status.get('total', 0)}, failed: {batch_status.get('failed', 0)}/{batch_status.get('total', 0)})")
    
    # Return the updated status
    return batch_status




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


@router.post("/video/clean_expired")
async def clean_expired_jobs(
    max_age: int = Query(300, description="Maximum age in seconds for jobs before cleaning"),
    video_service: VideoService = Depends(get_video_service)
):
    """
    Clean up single inference jobs that have been in the queue too long.
    
    Args:
        max_age: Maximum job age in seconds (default: 300 seconds/5 minutes)
    
    Returns:
        Dict with number of expired jobs cleaned up
    """
    try:
        # Create a method call to clean up the jobs
        async def clean_jobs():
            if not hasattr(video_service, "_clean_expired_jobs"):
                raise HTTPException(
                    status_code=500,
                    detail="Clean expired jobs function not available"
                )
            
            # We need to access the internal queue first and check size
            queue_size_before = video_service.job_queue.qsize()
            logger.info(f"Queue size before cleaning: {queue_size_before}")
            
            # Call the cleaning method
            await video_service._clean_expired_jobs(max_age)
            
            # Check queue size after
            queue_size_after = video_service.job_queue.qsize()
            logger.info(f"Queue size after cleaning: {queue_size_after}")
            
            return {
                "queue_size_before": queue_size_before,
                "queue_size_after": queue_size_after,
                "expired_jobs_removed": queue_size_before - queue_size_after
            }
            
        result = await clean_jobs()
        return result
    
    except Exception as e:
        logger.error(f"Error cleaning expired jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning expired jobs: {str(e)}")