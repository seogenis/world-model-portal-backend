# World Model Portal Backend

A high-performance backend for managing text-to-video generation using NVIDIA's Cosmos model with prompt tuning capabilities, asynchronous request handling, and multi-GPU batch processing.

## 🚀 Key Features

- **Video Generation**: Generate high-quality videos from text prompts using NVIDIA's Cosmos model
- **REST API Status Updates**: Efficient status tracking via stateless GET endpoints
- **Multi-GPU Batch Processing**: Generate multiple videos in parallel across 8 GPUs
- **Prompt Enhancement**: Convert basic concepts into detailed prompts
- **Parameter Extraction**: Identify key parameters from text descriptions
- **Prompt Updating**: Make targeted changes to prompts based on user requests
- **Prompt Variations**: Generate creative alternatives to explore different options
- **Task Queue with Worker Pool**: Non-blocking request handling with background processing
- **Rate Limiting**: Global semaphore to handle NVIDIA API rate limits
- **Persistent Session Management**: In-memory session storage with disk persistence across restarts
- **Filesystem-Based State Recovery**: Self-healing state recovery based on directory structure

## ⚙️ Architecture

The system consists of these main components:

1. **FastAPI Backend**: High-performance asynchronous API routes with RESTful endpoints
2. **Parameter Extractor**: Analyzes prompts to identify key parameters using OpenAI models
3. **Prompt Manager**: Tracks parameters, handles user modification requests, maintains prompt history
4. **Video Service**: Interfaces with NVIDIA's Cosmos API with retry logic and rate limiting
5. **Batch Inference Service**: Coordinates multi-GPU video generation with job distribution
6. **Worker Pool**: Background task processing for non-blocking operations
7. **State Recovery System**: Filesystem-based mechanism to recover status after restarts

## 🛠 Setup and Installation

### Prerequisites

- Python 3.10+
- NVIDIA API key for Cosmos model
- OpenAI API key for prompt enhancement features
- 8 GPUs for full batch processing capability (can run with fewer)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/world-model-portal-backend.git
   cd world-model-portal-backend
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NVIDIA_API_KEY=your_nvidia_api_key
   ```

### Running the Server

```bash
# Start the server (development mode)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the server (production mode)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using with ngrok for Public Access

To expose your server to the internet, you can use ngrok:

```bash
# Start ngrok (in another terminal)
./start_ngrok.sh
```

## 🔄 Asynchronous Processing

The backend uses an asynchronous processing architecture to handle long-running operations without blocking API responses:

1. **Request Queue**: Incoming requests are added to queues for processing
2. **Worker Pools**: Background workers process jobs from the queues
3. **Semaphore Rate Limiting**: Global semaphore controls NVIDIA API access (max 1 concurrent)
4. **Stateless Status Endpoints**: Efficient RESTful endpoints with filesystem validation
5. **Retry Mechanism**: Exponential backoff and retry for NVIDIA API rate limits
6. **Filesystem State Recovery**: Directory structure used to recover state after service restarts

## 🖥️ Key API Endpoints

### Video Generation

- `POST /api/video/single_inference`: Generate a video from a text prompt
- `GET /api/video/status/{job_id}`: Get video generation status with filesystem validation

### Batch Processing

- `POST /api/video/batch_inference`: Process multiple prompts in parallel (up to 8 GPUs)
- `GET /api/video/batch_status/{batch_id}`: Get batch status with comprehensive filesystem scanning
- `GET /api/video/batch_download/{batch_id}`: Download all batch videos as ZIP

### Prompt Management

- `POST /api/enhance`: Enhance a rough prompt with descriptive details
- `POST /api/initialize`: Extract parameters from a prompt
- `POST /api/update`: Update prompt based on user request
- `POST /api/generate-variations`: Generate variations of selected prompts
- `GET /api/parameters`: Get current parameters
- `GET /api/history`: Get prompt history

### Static Files

- `GET /api/videos/{video_id}`: Retrieve a generated video
- `GET /static/*`: Static files (HTML, videos, etc.)

## 🛠️ Key Components

### NVIDIA API Interaction

The system interacts with NVIDIA's Cosmos API for video generation:

```python
# Core interaction with rate limiting and retry logic
async with nvidia_api_semaphore:  # Global semaphore limits to 1 concurrent request
    # Send request to NVIDIA API with retry for rate limiting
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Send API request...
            if response.status_code == 429:  # Rate limited
                # Exponential backoff and retry
                await asyncio.sleep(retry_delay)
                retry_count += 1
                continue
            break  # Success
        except Exception as e:
            # Handle errors...
```

### Filesystem-Based Status Recovery

The system prioritizes filesystem evidence over in-memory status for robust state recovery:

```python
# Enhanced status endpoint with filesystem validation
@router.get("/video/status/{job_id}")
async def get_video_status(job_id: str, video_service: VideoService = Depends(get_video_service)):
    try:
        # First, check if the video file exists directly in the expected static path
        static_path = f"/static/videos/{job_id}/video.mp4"
        actual_file_path = Path(static_path.lstrip('/'))
        
        if actual_file_path.exists():
            # Video file exists at the expected location, job is complete
            logger.info(f"Found completed video file for job {job_id} at {static_path}")
            return VideoStatusResponse(
                job_id=job_id,
                status="complete",
                message=f"Video generation complete. Video available at: {static_path}",
                progress=100,
                video_url=static_path
            )
            
        # If not found, check other possible locations and states
        # (directory exists, legacy format, batch directory, etc.)
        
        # Only check in-memory status as last resort
        if hasattr(video_service, 'video_jobs') and job_id in video_service.video_jobs:
            # Return status from video_service if available
            status_data = video_service.video_jobs[job_id]
            return VideoStatusResponse(...)
            
        # Default fallback response if no state can be determined
        return VideoStatusResponse(
            job_id=job_id,
            status="pending",
            message=f"Job is queued or not found. Expected path when complete: {static_path}",
            progress=10,
            video_url=None
        )
    except Exception as e:
        # Handle errors...
```

### Worker Pool for Background Processing

Task queues and worker pools handle background processing without blocking:

```python
# Worker pool implementation
async def _job_worker(self):
    while True:
        try:
            # Get next job from queue with timeout
            job_id, prompt, websocket = await self.job_queue.get()
            
            # Process job asynchronously
            task = asyncio.create_task(
                self._process_video_generation(job_id, prompt, websocket)
            )
            
            # Continue processing other jobs
        except Exception as e:
            # Handle worker errors...
```

### Multi-GPU Batch Inference

Batch processing distributes jobs across multiple GPUs:

```python
# Distribute jobs across GPUs
for i, prompt in enumerate(prompts):
    job_id = f"job_{i}"
    gpu_id = i % self.num_gpus  # Assign to GPU
    
    # Start job on this GPU
    task = asyncio.create_task(
        self._process_single_job(batch_id, job_id, prompt, job_dir, gpu_id, websocket)
    )
```

## 📂 Project Structure

```
.
├── app/
│   ├── api/                 # API routes and schemas
│   │   ├── routes.py        # FastAPI routes and REST endpoints
│   │   └── schemas.py       # Pydantic models for API
│   ├── core/                # Core configuration and utilities
│   │   ├── config.py        # Application settings and semaphore
│   │   └── logger.py        # Logging configuration
│   ├── services/            # Business logic services
│   │   ├── batch_inference_service.py  # Multi-GPU batch processing
│   │   ├── parameter_extractor.py      # Extract parameters from prompts
│   │   ├── prompt_manager.py           # Manage prompt refinement
│   │   ├── session_manager.py          # Session persistence with disk storage
│   │   └── video_service.py            # NVIDIA API interface with rate limiting
│   ├── utils/               # Helper utilities
│   │   └── openai_utils.py  # OpenAI API helpers
│   └── main.py              # FastAPI application entry point
├── sessions/                # Persistent session storage directory
├── static/                  # Static files
│   ├── frontend_examples/   # Example HTML interfaces
│   │   └── demo_complete.html  # Comprehensive demo UI
│   └── videos/              # Generated videos storage
├── API_DOCUMENTATION.md     # API documentation
├── README.md                # Project documentation
├── user_flow_instructions.txt # Frontend integration guide
├── start_ngrok.sh           # Script to start ngrok
└── requirements.txt         # Python dependencies
```

## 🧩 Implementation Details

### Recent Improvements

1. **Rate Limiting & Retry Handling**: 
   - Added global semaphore for NVIDIA API access across services
   - Implemented exponential backoff retry logic
   - Added proper error handling for 429 rate limit responses

2. **Robust Status Updates**:
   - Enhanced REST API endpoints with filesystem validation
   - Added self-healing state recovery from directory structure
   - Improved cross-referencing between job and batch IDs

3. **Asynchronous Processing Architecture**:
   - Implemented proper worker pools for background processing
   - Added job queues for managing video generation tasks
   - Fixed "no running event loop" issues in async initialization

4. **Persistent Session Management**:
   - Implemented disk-based session persistence across restarts
   - Added singleton pattern to prevent multiple instances losing sessions
   - Implemented efficient loading/cleanup of sessions from disk

5. **State Recovery System**:
   - Added filesystem scanning to recover job status after restarts
   - Implemented hierarchical directory structure for batch and job organization
   - Enhanced error handling with filesystem evidence prioritization

### Video Generation Flow

1. Client submits a prompt through the API
2. System generates a unique job ID and adds to job queue
3. Background worker picks up the job when resources are available
4. Worker acquires semaphore for NVIDIA API access (rate limiting)
5. Request is sent to NVIDIA Cosmos API with retry logic for rate limits
6. Client polls status endpoint for updates on generation progress
7. Generated video is extracted and made available via static URL
8. Client displays video when generation is complete

### Design Concept from Original Blueprint

This implementation draws inspiration from a comprehensive design for a prompt-tuning agent for NVIDIA's Cosmos text-to-video model. Key concepts adapted from the original design include:

1. **Parameter Extraction**: Using LLMs to extract structured parameters from natural language prompts
2. **Prompt Refinement**: Allowing users to modify prompts through conversational interaction
3. **Modular Architecture**: Separating concerns into distinct services with clean interfaces
4. **Asynchronous Processing**: Non-blocking operations for responsive user experience
5. **Optimization Strategies**: Model selection and caching for performance enhancement
6. **Self-Healing Systems**: Robust recovery mechanisms to maintain state after restarts

### Batch Video Generation

For high-throughput scenarios, the system supports parallel batch generation:

1. Submit up to 8 prompts via batch endpoint
2. Each prompt is assigned to a different GPU (0-7) for concurrent processing
3. All jobs share the same global semaphore for NVIDIA API access
4. Status updates are available via the batch_status endpoint
5. All generated videos can be downloaded as a single ZIP file
6. Directory structure preserves job organization even after service restarts

## 🚀 Deploying on Brev.dev

This project is optimized for deployment on a Brev.dev instance with 8 GPUs:

1. Create a Brev instance with 8 GPUs
2. Clone the repository and install dependencies
3. Set up environment variables with API keys
4. Start the server with uvicorn
5. (Optional) Use ngrok for public access

## 📊 Performance Considerations

- **NVIDIA API Rate Limits**: The Cosmos API limits concurrent requests to 1 per API key
- **GPU Memory Usage**: Each video generation job consumes significant GPU memory
- **Worker Pool Sizing**: The worker pool is configured to handle multiple queued jobs
- **Filesystem I/O**: Status endpoints scan directories for accurate state recovery
- **Client Polling Strategy**: Clients should implement exponential backoff for polling

## 🔮 Future Enhancements

1. **Docker Containerization**: Containerize for easy deployment and scaling
2. **Authentication System**: Add API keys and user management
3. **Advanced Monitoring**: Prometheus metrics and Grafana dashboards
4. **Custom Prompt Templates**: User-defined prompt templates and styles
5. **Enhanced State Persistence**: Improved file-based persistence for high-reliability
6. **Distributed Storage**: Add optional distributed storage for high-availability deployments

## License

This project is licensed under the MIT License - see the LICENSE file for details.