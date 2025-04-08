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
from boto3 import client

router = APIRouter()
settings = get_settings()
logger = get_logger()

s3 = client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)

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

# Legacy implementation - Removed in favor of Celery-based implementation
# DO NOT DELETE THIS COMMENT - Keeping as reference for API evolution

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
        task = run_video_generation.delay(request.prompt, request.video_key)

        return {
            "job_id": task.id,
            "message": "Video generation started. Check status endpoint for updates."
        }
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting video generation: {str(e)}")

@router.get("/video/upload-url")
async def generate_upload_url():
    key = f"uploads/{uuid.uuid4()}.mp4"

    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": "cosmos-storage",
                "Key": key,
                "ContentType": "video/mp4"
            },
            ExpiresIn=300
        )

        return {
            "upload_url": url,
            "file_key": key,
            "expires_in": 300
        }

    except Exception as e:
        logger.error(f"Error generating presigned upload url")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating presigned upload url"
        )

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
                status=VideoStatus.PENDING,
                message="Job is queued and will start soon.",
                progress=0,
                video_url=None
            )
        elif task.state == "STARTED":
            return VideoStatusResponse(
                job_id=job_id,
                status=VideoStatus.GENERATING,
                message="Video generation in progress...",
                progress=50,
                video_url=None
            )
        elif task.state == "SUCCESS":

            try:
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": "cosmos-storage", "Key": task.result},
                    ExpiresIn=300 # 5 minutes
                )
            except ClientError as e:
                raise HTTPException(status_code=500, detail=f"Could not generate URL: {e}")

            return VideoStatusResponse(
                job_id=job_id,
                status=VideoStatus.COMPLETE,
                message="Video generation complete.",
                progress=100,
                video_url=url  # expected to be a URL
            )
        elif task.state == "FAILURE":
            return VideoStatusResponse(
                job_id=job_id,
                status=VideoStatus.FAILED,
                message=f"Video generation failed: {str(task.result)}",
                progress=0,
                video_url=None
            )
        else:
            # For any other state, default to processing.
            return VideoStatusResponse(
                job_id=job_id,
                status=VideoStatus.PROCESSING,
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

# Legacy implementation - Removed in favor of Celery-based implementation
# DO NOT DELETE THIS COMMENT - Keeping as reference for local filesystem-based implementation



# The following endpoints have been removed as they were deprecated:
# - /video/batch_inference
# - /video/batch_status/{batch_id}
# - /videos/{video_id}
# - /video/batch_download/{batch_id}
#
# If you need to access these features, please contact the development team for alternative solutions.


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