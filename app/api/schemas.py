from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class InitializeRequest(BaseModel):
    """Request model for initializing a prompt."""
    prompt: str = Field(..., description="The initial text-to-video prompt")


class UpdateRequest(BaseModel):
    """Request model for updating a prompt."""
    user_request: str = Field(..., description="The user's request to modify the prompt")


class PromptResponse(BaseModel):
    """Response model for prompt operations."""
    parameters: Dict[str, Any] = Field(..., description="Extracted parameters from the prompt")
    prompt: str = Field(..., description="The current prompt text")
    changes: List[str] = Field(default=[], description="List of changes made in this update")


class EnhancePromptRequest(BaseModel):
    """Request model for enhancing a rough prompt."""
    rough_prompt: str = Field(..., description="The rough prompt to enhance with details")


class EnhancePromptResponse(BaseModel):
    """Response model for enhanced prompts."""
    original_prompt: str = Field(..., description="The original rough prompt")
    enhanced_prompt: str = Field(..., description="The enhanced prompt with descriptive details")


class GenerateVariationsRequest(BaseModel):
    """Request model for generating prompt variations."""
    selected_indices: List[int] = Field(..., description="Indices of selected prompts from prompt history")
    total_count: int = Field(8, description="Total number of prompts to generate (including selected ones)")
    
    @validator('total_count')
    def total_count_must_be_reasonable(cls, v):
        if v < 1 or v > 20:
            raise ValueError('total_count must be between 1 and 20')
        return v
    
    @validator('selected_indices')
    def selected_indices_must_be_non_empty(cls, v):
        if not v:
            raise ValueError('At least one prompt index must be selected')
        return v


class GenerateVariationsResponse(BaseModel):
    """Response model for prompt variations."""
    prompts: List[str] = Field(..., description="List of prompt variations")
    selected_indices: List[int] = Field(..., description="The indices that were selected")


class PromptHistoryItem(BaseModel):
    """Model for a prompt history item."""
    prompt: str = Field(..., description="The prompt text")
    parameters: Dict[str, Any] = Field(..., description="Parameters for this prompt")
    description: str = Field(..., description="Description of this prompt version")


class PromptHistoryResponse(BaseModel):
    """Response model for prompt history."""
    history: List[PromptHistoryItem] = Field(..., description="List of all prompts")


class VideoStatus(str, Enum):
    """Status of video generation process."""
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoGenerationRequest(BaseModel):
    """Request model for generating a video from a prompt."""
    prompt: str = Field(..., description="The text-to-video prompt")
    
    
class BatchVideoGenerationRequest(BaseModel):
    """Request model for generating multiple videos from prompts."""
    prompts: List[str] = Field(..., description="List of text-to-video prompts")
    
    @validator('prompts')
    def prompts_must_be_reasonable(cls, v):
        if not v:
            raise ValueError('At least one prompt must be provided')
        if len(v) > 10:
            raise ValueError('Maximum 10 prompts allowed in a batch')
        return v


class VideoStatusResponse(BaseModel):
    """Response model for video generation status."""
    job_id: str = Field(..., description="Unique identifier for the video generation job")
    status: VideoStatus = Field(..., description="Current status of the video generation")
    message: str = Field("", description="Additional status message or error details")
    progress: int = Field(0, description="Progress percentage (0-100)")
    video_url: Optional[str] = Field(None, description="URL to the generated video if complete")


class BatchVideoStatusResponse(BaseModel):
    """Response model for batch video generation status."""
    job_ids: List[str] = Field(..., description="List of job ids for each video in the batch")
    statuses: List[VideoStatusResponse] = Field(..., description="Status information for each video")


