import logging
import asyncio
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import atexit
from datetime import datetime

from app.api.routes import router as api_router
from app.core.config import get_settings, update_nvidia_semaphore
from app.core.logger import get_logger

# Create app
app = FastAPI(title=get_settings().PROJECT_NAME, version=get_settings().VERSION)

# Setup logging
logger = get_logger()

# Configure routes
api_router_with_prefix = APIRouter(prefix=get_settings().API_PREFIX)
api_router_with_prefix.include_router(api_router)
app.include_router(api_router_with_prefix)

# Configure static files
os.makedirs("static", exist_ok=True)
os.makedirs("static/videos", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task for session cleanup
session_cleanup_task = None
cleanup_interval = 3600  # 1 hour

# Create service instances at app start to reuse throughout
parameter_extractor_instance = None
session_manager_instance = None

def get_global_session_manager():
    """Get a singleton instance of the SessionManagerService."""
    global parameter_extractor_instance, session_manager_instance
    
    if session_manager_instance is None:
        from app.services.parameter_extractor import ParameterExtractor
        from app.services.session_manager import SessionManagerService
        
        parameter_extractor_instance = ParameterExtractor()
        session_manager_instance = SessionManagerService(parameter_extractor_instance)
        
    return session_manager_instance

async def run_session_cleanup():
    """
    Background task to periodically clean up expired sessions.
    """
    logger.info("Starting session cleanup background task")
    
    while True:
        try:
            # Get the global SessionManagerService instance
            session_manager = get_global_session_manager()
            
            # Cleanup expired sessions
            removed = await session_manager.cleanup_expired_sessions()
            logger.info(f"Session cleanup removed {removed} expired sessions")
            
            # Sleep for the configured interval
            await asyncio.sleep(cleanup_interval)
            
        except asyncio.CancelledError:
            # Task is being cancelled, clean up and exit
            logger.info("Session cleanup task cancelled")
            break
        except Exception as e:
            # Log the error and continue
            logger.error(f"Error in session cleanup task: {str(e)}")
            await asyncio.sleep(60)  # Sleep for a minute before retrying

@app.on_event("startup")
async def startup_event():
    # Configure NVIDIA API semaphore
    logger.info(f"Configured NVIDIA API concurrency to {get_settings().NVIDIA_MAX_CONCURRENT}")
    update_nvidia_semaphore(get_settings().NVIDIA_MAX_CONCURRENT)
    
    # Start the session cleanup task
    global session_cleanup_task
    session_cleanup_task = asyncio.create_task(run_session_cleanup())
    logger.info("Session cleanup background task started")

@app.on_event("shutdown")
async def shutdown_event():
    # Cancel the session cleanup task
    if session_cleanup_task:
        session_cleanup_task.cancel()
        try:
            await session_cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Session cleanup background task stopped")

@app.get("/")
async def root():
    return {"message": "Welcome to the Cosmos Prompt Tuner API", "version": get_settings().VERSION}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": get_settings().VERSION
    }