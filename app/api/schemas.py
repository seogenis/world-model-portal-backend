from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from uuid import UUID
from typing import Optional


class SessionRequest(BaseModel):
    """Request model with optional session ID."""
    session_id: Optional[str] = Field(None, description="Optional session ID for persistent state")
    
class SessionIdRequest(BaseModel):
    """Request model with required session ID."""
    session_id: str = Field(..., description="Session ID for accessing stored state")


class InitializeRequest(SessionRequest):
    """Request model for initializing a prompt."""
    prompt: str = Field(..., description="The initial text-to-video prompt")


class UpdateRequest(SessionRequest):
    """Request model for updating a prompt."""
    user_request: str = Field(..., description="The user's request to modify the prompt")


class PromptResponse(BaseModel):
    """Response model for prompt operations."""
    parameters: Dict[str, Any] = Field(..., description="Extracted parameters from the prompt")
    prompt: str = Field(..., description="The current prompt text")
    changes: List[str] = Field(default=[], description="List of changes made in this update")
    session_id: str = Field(..., description="Session ID for this prompt state")


class EnhancePromptRequest(SessionRequest):
    """Request model for enhancing a rough prompt."""
    rough_prompt: str = Field(..., description="The rough prompt to enhance with details")


class EnhancePromptResponse(BaseModel):
    """Response model for enhanced prompts."""
    original_prompt: str = Field(..., description="The original rough prompt")
    enhanced_prompt: str = Field(..., description="The enhanced prompt with descriptive details")
    session_id: str = Field(..., description="Session ID for this prompt state")


class GenerateVariationsRequest(SessionRequest):
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
    session_id: str = Field(..., description="Session ID for this prompt state")


class PromptHistoryItem(BaseModel):
    """Model for a prompt history item."""
    prompt: str = Field(..., description="The prompt text")
    parameters: Dict[str, Any] = Field(..., description="Parameters for this prompt")
    description: str = Field(..., description="Description of this prompt version")


class PromptHistoryResponse(BaseModel):
    """Response model for prompt history."""
    history: List[PromptHistoryItem] = Field(..., description="List of all prompts")
    session_id: str = Field(..., description="Session ID for this prompt state")


class VideoStatus(str, Enum):
    """Status of video generation process."""
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    EXPIRED = "expired"


class VideoGenerationRequest(BaseModel):
    """Request model for generating a video from a prompt."""
    prompt: str = Field(..., description="The text-to-video prompt")
    video_key: Optional[str] = Field(default=None, description="The file key for optional video upload")
    
    
class BatchVideoGenerationRequest(BaseModel):
    """Request model for generating multiple videos from prompts."""
    prompts: List[str] = Field(..., description="List of text-to-video prompts")
    
    @validator('prompts')
    def prompts_must_be_reasonable(cls, v):
        if not v:
            raise ValueError('At least one prompt must be provided')
        if len(v) > 8:
            raise ValueError('Maximum 8 prompts allowed in a batch')
        return v


class VideoStatusResponse(BaseModel):
    """Response model for video generation status."""
    job_id: str = Field(..., description="Unique identifier for the video generation job")
    status: VideoStatus = Field(..., description="Current status of the video generation")
    message: str = Field("", description="Additional status message or error details")
    progress: int = Field(0, description="Progress percentage (0-100)")
    video_url: Optional[str] = Field(None, description="URL to the generated video if complete")


class BatchJobStatus(BaseModel):
    """Status information for a single job in a batch."""
    job_id: str = Field(..., description="Unique identifier for this specific job")
    status: VideoStatus = Field(..., description="Current status of the video generation")
    message: str = Field("", description="Status message or error details")
    progress: int = Field(0, description="Progress percentage (0-100)")
    video_url: Optional[str] = Field(None, description="URL to the generated video if complete")
    gpu_id: int = Field(..., description="GPU ID used for this job")
    prompt: str = Field(..., description="The prompt used for this job")


class BatchStatus(str, Enum):
    """Status of batch processing."""
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"


class BatchVideoStatusResponse(BaseModel):
    """Response model for batch video generation status."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    status: str = Field(..., description="Overall batch status (processing, complete, failed, partial)")
    total: int = Field(..., description="Total number of jobs in the batch")
    completed: int = Field(0, description="Number of completed jobs")
    failed: int = Field(0, description="Number of failed jobs")
    message: Optional[str] = Field(None, description="Overall status message")
    jobs: List[BatchJobStatus] = Field(..., description="Status information for each job in the batch")
