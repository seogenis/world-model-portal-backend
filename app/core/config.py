from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


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
    
    # Redis Cache Configuration (optional)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # Cache TTL in seconds
    
    # Model Selection
    PARAMETER_EXTRACTION_MODEL: str = "gpt-4o-mini"
    UPDATE_REQUEST_MODEL: str = "gpt-4o-mini"
    PROMPT_REGENERATION_MODEL: str = "gpt-4o-mini"
    PROMPT_ENHANCEMENT_MODEL: str = "gpt-4o-mini"
    PROMPT_VARIATION_MODEL: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()