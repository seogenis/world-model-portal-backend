from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from app.api.routes import router as api_router
from app.core.config import get_settings
import os
from pathlib import Path

settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    Prompt tuning agent for NVIDIA Cosmos text-to-video model.
    
    ## API Highlights
    
    - **Single Video Generation**: Generate videos from text prompts on a single GPU
    - **Batch Video Generation**: Process multiple prompts in parallel (up to 8 GPUs)
    - **Prompt Management**: Fine-tune and enhance prompts for better results
    - **Real-time Updates**: Get live progress via WebSockets
    
    ## Quick Links
    
    - [API Documentation](/api_docs) - Detailed API documentation
    - [Demo Interface](/demo) - Interactive demo to test the API
    """,
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
os.makedirs("static/frontend_examples", exist_ok=True)

# Copy frontend examples and API documentation to static directory for easy access
project_root = Path(__file__).parent.parent.parent

# List of frontend examples to copy
frontend_examples = [
    "video_generator_complete.html",  # Main comprehensive example
    "demo_complete.html",             # Complete demo with all sections
    "polling_example.html",           # Minimal polling example
    "batch_video_generator.html"      # Legacy example
]

# Copy each example
for example in frontend_examples:
    source_html = project_root / "frontend_examples" / example
    dest_html = project_root / "static" / "frontend_examples" / example
    if source_html.exists():
        with open(source_html, "r") as src, open(dest_html, "w") as dst:
            dst.write(src.read())

source_docs = project_root / "API_DOCUMENTATION.md"
dest_docs = project_root / "static" / "API_DOCUMENTATION.md"
if source_docs.exists():
    with open(source_docs, "r") as src, open(dest_docs, "w") as dst:
        dst.write(src.read())

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
        "example_app": "/demo",
        "api_docs": "/api_docs",
    }


@app.get("/demo", response_class=HTMLResponse)
async def video_demo():
    """Serve the video generation demo page."""
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/static/frontend_examples/demo_complete.html" />
</head>
<body>
    <p>Redirecting to demo...</p>
</body>
</html>"""
    )


@app.get("/api_docs")
async def api_documentation():
    """Serve the API documentation markdown file."""
    return FileResponse(
        path="static/API_DOCUMENTATION.md",
        media_type="text/markdown",
        filename="API_DOCUMENTATION.md"
    )