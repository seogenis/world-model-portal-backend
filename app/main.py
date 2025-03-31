from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.core.config import get_settings
import os
from pathlib import Path

settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Prompt tuning agent for NVIDIA Cosmos text-to-video model",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)

# Create static directory for videos if it doesn't exist
os.makedirs("static/videos", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint providing basic info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "Prompt tuning agent for NVIDIA Cosmos text-to-video model",
        "docs": "/docs",
    }


@app.get("/api/video")
async def get_video():
    """Serve the video file from the root directory."""
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "rickroll-roll.mp4"
    
    if not video_path.exists():
        return {"error": "Video file not found"}
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename="video.mp4"
    )


@app.get("/api/video/stream")
async def stream_video():
    """Stream the default video file from the root directory."""
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "rickroll-roll.mp4"
    
    if not video_path.exists():
        return {"error": "Video file not found"}
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4"
    )


@app.get("/api/video/stream/{filename}")
async def stream_video_by_filename(filename: str):
    """Stream a specific video file by filename."""
    project_root = Path(__file__).parent.parent.parent
    
    # Ensure the filename doesn't contain path traversal attempts
    safe_filename = os.path.basename(filename)
    
    # Add .mp4 extension if not present
    if not safe_filename.endswith('.mp4'):
        safe_filename += '.mp4'
    
    video_path = project_root / safe_filename
    
    if not video_path.exists():
        return {"error": f"Video file '{safe_filename}' not found"}
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4"
    )


@app.get("/api/video/{filename}")
async def get_video_by_filename(filename: str):
    """Serve a specific video file by filename."""
    project_root = Path(__file__).parent.parent.parent
    
    # Ensure the filename doesn't contain path traversal attempts
    safe_filename = os.path.basename(filename)
    
    # Add .mp4 extension if not present
    if not safe_filename.endswith('.mp4'):
        safe_filename += '.mp4'
    
    video_path = project_root / safe_filename
    
    if not video_path.exists():
        return {"error": f"Video file '{safe_filename}' not found"}
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=safe_filename
    )


@app.get("/api/videos")
async def list_videos():
    """List all available video files in the root directory."""
    project_root = Path(__file__).parent.parent.parent
    
    # Get all mp4 files in the root directory
    mp4_files = list(project_root.glob("*.mp4"))
    
    videos = []
    for file_path in mp4_files:
        file_name = file_path.name
        file_size = file_path.stat().st_size
        videos.append({
            "filename": file_name,
            "url": f"/api/video/{file_name}",
            "stream_url": f"/api/video/stream/{file_name}",
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)  # Convert to MB
        })
    
    return {"videos": videos}