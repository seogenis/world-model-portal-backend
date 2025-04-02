from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import asyncio


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # API Configuration
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    PROJECT_NAME: str = "Cosmos Prompt Tuner"
    VERSION: str = "0.1.0"
    
    # API Keys Configuration
    OPENAI_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    NVIDIA_API_KEY_BACKUP1: str = ""
    NVIDIA_API_KEY_BACKUP2: str = ""
    
    # Session configuration
    CACHE_TTL: int = 3600  # Session TTL in seconds (1 hour)
    
    # No external database used - sessions stored on filesystem
    
    # Model Selection
    PARAMETER_EXTRACTION_MODEL: str = "gpt-4o-mini"
    UPDATE_REQUEST_MODEL: str = "gpt-4o-mini"
    PROMPT_REGENERATION_MODEL: str = "gpt-4o-mini"
    PROMPT_ENHANCEMENT_MODEL: str = "gpt-4o-mini"
    PROMPT_VARIATION_MODEL: str = "gpt-4o-mini"
    
    # NVIDIA API Rate Limiting
    NVIDIA_MAX_CONCURRENT: int = 1  # Maximum concurrent NVIDIA API calls
    NVIDIA_RETRY_DELAY: int = 5  # Seconds to wait before retrying after rate limit
    NVIDIA_RETRY_ATTEMPTS: int = 3  # Maximum number of retry attempts
    ENABLE_API_KEY_ROTATION: bool = True  # Whether to rotate API keys on errors
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global semaphore to control NVIDIA API access
nvidia_api_semaphore = asyncio.Semaphore(1)  # Default to 1 concurrent request

def update_nvidia_semaphore(max_concurrent: int):
    """Update the NVIDIA API semaphore limit"""
    global nvidia_api_semaphore
    # Create a new semaphore with the updated limit
    nvidia_api_semaphore = asyncio.Semaphore(max_concurrent)



@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    from app.core.logger import get_logger
    logger = get_logger()
    
    settings = Settings()
    
    # Log API key status
    if not settings.NVIDIA_API_KEY:
        logger.warning("NVIDIA_API_KEY is not set in environment variables or .env file")
        logger.info("Local GPU fallback is enabled - will use local GPUs when NVIDIA API fails")
    
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set in environment variables or .env file")
    
    # Update semaphore based on settings
    update_nvidia_semaphore(settings.NVIDIA_MAX_CONCURRENT)
    
    return settings