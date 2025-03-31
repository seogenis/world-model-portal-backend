# World Model Portal API Documentation

This document describes the available API endpoints for interacting with the World Model Portal backend, which provides prompt tuning and video generation services using NVIDIA's Cosmos text-to-video model.

## Base URL

All API endpoints are prefixed with `/api`.

## Video Generation Endpoints

### 1. Generate Single Video

**Endpoint:** `POST /api/video/single_inference`

Generates a single video from a text prompt.

**Request:**
```json
{
  "prompt": "A beautiful sunrise over a mountain lake with reflections in the water"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Video generation started. Connect to WebSocket for updates."
}
```

### 2. Get Single Video Status

**Endpoint:** `GET /api/video/status/{job_id}`

Get the current status of a single video generation job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "generating",
  "message": "Generating video from prompt",
  "progress": 65,
  "video_url": null
}
```

Status can be one of: `pending`, `generating`, `processing`, `complete`, or `failed`.

### 3. Generate Multiple Videos (Batch)

**Endpoint:** `POST /api/video/batch_inference`

Generates multiple videos in parallel, each on a different GPU.

**Request:**
```json
{
  "prompts": [
    "A sunset over the ocean with waves crashing on the shore",
    "A bustling city street at night with neon lights",
    "A spaceship landing on an alien planet",
    "A forest with sunlight streaming through the trees"
  ]
}
```

**Response:**
```json
{
  "batch_id": "662e8400-e29b-41d4-a716-446655440123",
  "message": "Batch processing started for 4 prompts. Connect to WebSocket for updates."
}
```

### 4. Get Batch Status

**Endpoint:** `GET /api/video/batch_status/{batch_id}`

Get the current status of all videos in a batch.

**Response:**
```json
{
  "batch_id": "662e8400-e29b-41d4-a716-446655440123",
  "status": "processing",
  "total": 4,
  "completed": 1,
  "failed": 0,
  "message": "Completed: 1/4",
  "jobs": [
    {
      "job_id": "job_0",
      "status": "complete",
      "message": "Video generation complete",
      "progress": 100,
      "video_url": "/static/videos/662e8400-e29b-41d4-a716-446655440123/job_0/job_0.mp4",
      "gpu_id": 0,
      "prompt": "A sunset over the ocean with waves crashing on the shore"
    },
    {
      "job_id": "job_1",
      "status": "generating",
      "message": "Generating video from prompt",
      "progress": 75,
      "video_url": null,
      "gpu_id": 1,
      "prompt": "A bustling city street at night with neon lights"
    },
    // ...more jobs
  ]
}
```

Batch status can be one of: `processing`, `complete`, `partial`, `failed`, or `not_found`.

### 5. Download Batch Videos as ZIP

**Endpoint:** `GET /api/video/batch_download/{batch_id}`

Download all successfully generated videos from a batch as a ZIP file.

**Response:** A ZIP file containing all generated videos.

### 6. Retrieve Video by ID

**Endpoint:** `GET /api/videos/{video_id}`

Unified endpoint to retrieve any video by its ID (job_id or filename).

**Response:** The video file (MP4).

## Status Polling Approach

For the most reliable operation, we recommend using a polling approach instead of WebSockets to check video generation status.

### 1. Single Video Status Polling

**Endpoint:** `GET /api/video/status/{job_id}`

Poll this endpoint at regular intervals (e.g., every 2-3 seconds) to get updates about a single video generation job.

**Response Format:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "generating",
  "message": "Generating video...",
  "progress": 45,
  "video_url": null
}
```

When the video is complete, you'll receive a response with the video URL:
```json
{
  "status": "complete",
  "message": "Video generation complete",
  "progress": 100,
  "video_url": "/api/videos/550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. Batch Status Polling

**Endpoint:** `GET /api/video/batch_status/{batch_id}`

Poll this endpoint at regular intervals (e.g., every 3-5 seconds) to get updates about a batch video generation job.

**Response Format:**
```json
{
  "batch_id": "662e8400-e29b-41d4-a716-446655440123",
  "status": "processing",
  "total": 4,
  "completed": 2,
  "failed": 0,
  "message": "Completed: 2/4",
  "jobs": [
    // Array of job status objects (same as in batch status endpoint)
  ]
}
```

## WebSocket Endpoints (Deprecated)

WebSocket endpoints are still available but not recommended due to reliability issues. Use polling approach instead.

### 1. Single Video Status Updates

**Endpoint:** `websocket /api/ws/video/{job_id}`

### 2. Batch Status Updates

**Endpoint:** `websocket /api/ws/batch/{batch_id}`

## Prompt Management Endpoints

### 1. Initialize Prompt

**Endpoint:** `POST /api/initialize`

Initialize the system with a new prompt for refinement.

**Request:**
```json
{
  "prompt": "A colorful sunset over the ocean"
}
```

**Response:**
```json
{
  "parameters": {
    "time": "sunset",
    "scene": "ocean",
    "colors": ["orange", "red", "purple"],
    // other extracted parameters
  },
  "prompt": "A colorful sunset over the ocean",
  "changes": []
}
```

### 2. Update Prompt

**Endpoint:** `POST /api/update`

Update the prompt based on a natural language request.

**Request:**
```json
{
  "user_request": "Make it more dramatic with stormy clouds"
}
```

**Response:**
```json
{
  "parameters": {
    "time": "sunset",
    "scene": "ocean",
    "weather": "stormy",
    "sky": "dramatic clouds",
    // updated parameters
  },
  "prompt": "A dramatic sunset over the stormy ocean with dark clouds gathering on the horizon",
  "changes": [
    "Added stormy weather",
    "Made sky more dramatic with clouds"
  ]
}
```

### 3. Enhance Prompt

**Endpoint:** `POST /api/enhance`

Enhance a basic prompt with more descriptive details.

**Request:**
```json
{
  "rough_prompt": "cats playing"
}
```

**Response:**
```json
{
  "original_prompt": "cats playing",
  "enhanced_prompt": "Two fluffy tabby cats playfully chasing each other through a sunlit living room, their paws batting at colorful toy mice"
}
```

### 4. Generate Variations

**Endpoint:** `POST /api/generate-variations`

Generate variations of selected prompts from history.

**Request:**
```json
{
  "selected_indices": [0, 2],
  "total_count": 3
}
```

**Response:**
```json
{
  "prompts": [
    "A dramatic sunset over the stormy ocean with lightning strikes illuminating dark clouds",
    "A peaceful sunset over calm ocean waters with gentle waves and seagulls flying overhead",
    "A vibrant sunset over a tropical ocean with palm trees silhouetted against the colorful sky"
  ],
  "selected_indices": [0, 2]
}
```

### 5. Get Prompt History

**Endpoint:** `GET /api/history`

Get the history of all prompts created in the current session.

**Response:**
```json
{
  "history": [
    {
      "prompt": "A colorful sunset over the ocean",
      "parameters": {
        // parameters object
      },
      "description": "Initial prompt"
    },
    {
      "prompt": "A dramatic sunset over the stormy ocean with dark clouds gathering on the horizon",
      "parameters": {
        // parameters object
      },
      "description": "Added storm effects"
    }
  ]
}
```

### 6. Get Current Parameters

**Endpoint:** `GET /api/parameters`

Get the current parameters extracted from the active prompt.

**Response:**
```json
{
  "time": "sunset",
  "scene": "ocean",
  "weather": "stormy",
  "sky": "dark clouds",
  // all current parameters
}
```

## Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Notes for Frontend Developers

1. **Video Generation Flow:**
   - Start a video generation job (single or batch)
   - Connect to corresponding WebSocket for real-time updates
   - Once complete, use the video_url to display the video or download it

2. **Batch Processing:**
   - Limited to 8 prompts (one per GPU)
   - Each prompt runs on a separate GPU
   - All videos can be downloaded as a ZIP when complete

3. **Video URLs:**
   - Videos are served from `/static/videos/` path
   - Use the unified `/api/videos/{video_id}` endpoint to retrieve videos