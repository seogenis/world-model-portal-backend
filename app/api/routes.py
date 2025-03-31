from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import os
import asyncio

from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager
from app.services.video_service import VideoService
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
    VideoStatusResponse,
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


@router.websocket("/ws/video/{job_id}")
async def video_status_websocket(websocket: WebSocket, job_id: str, video_service: VideoService = Depends(get_video_service)):
    """
    WebSocket connection for real-time video generation status updates.
    
    Clients should connect to this endpoint after initiating a video generation job.
    The server will send updates about the job status until it completes or fails.
    """
    await websocket.accept()
    
    try:
        # Check if the job exists in the video_jobs dictionary
        if job_id in video_service.video_jobs:
            # If it exists, send the current status
            await websocket.send_json(video_service.video_jobs[job_id])
        else:
            # Otherwise, wait for updates from the video generation process
            # The video service will handle sending updates through this WebSocket
            while True:
                # Keep the connection open
                await asyncio.sleep(1)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for job {job_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        await websocket.close()