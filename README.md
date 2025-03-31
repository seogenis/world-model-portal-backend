# Cosmos Text2World Prompt Tuning API

An intelligent prompt tuning agent for NVIDIA's Cosmos text-to-video model that helps users refine text prompts and generate videos through intuitive interactions.

## Features

- **Prompt Enhancement**: Convert rough ideas into detailed prompts
- **Parameter Extraction**: Identify key parameters from text descriptions
- **Prompt Updating**: Make minimal, focused changes to prompts based on user requests
- **Prompt Variations**: Generate creative alternatives to explore different options
- **Video Generation**: Generate videos using NVIDIA's Cosmos API
- **Status Updates**: Reliable status polling for generation progress
- **Multi-GPU Batch Inference**: Generate multiple videos simultaneously across multiple GPUs

## Architecture

The system consists of these main components:

1. **Parameter Extractor**: Analyzes prompts to identify key parameters
2. **Prompt Manager**: Tracks parameters, handles user modification requests, and manages prompt history
3. **Prompt Enhancer**: Transforms rough prompts into detailed, descriptive content 
4. **Variation Generator**: Creates alternative versions of prompts based on selected examples
5. **Video Service**: Interfaces with NVIDIA's Cosmos API to generate videos
6. **API Layer**: Exposes functionality through REST endpoints with WebSocket support

## Setup and Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/cosmos-prompt-tuner.git
   cd cosmos-prompt-tuner
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NVIDIA_API_KEY=your_nvidia_api_key
   ```

## Running the Server

```bash
# Start the server (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the server (production)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Using with ngrok

To expose your server to the internet, you can use ngrok:

```bash
# Start ngrok (in another terminal)
./start_ngrok.sh
```

## API Endpoints

### Prompt Management

- `POST /api/enhance`: Enhance a rough prompt with descriptive details
- `POST /api/initialize`: Initialize a new prompt
- `POST /api/update`: Update parameters based on user request
- `GET /api/history`: Get history of all prompts
- `POST /api/generate-variations`: Generate variations of selected prompts
- `GET /api/parameters`: Get current parameters

### Video Generation

- `POST /api/video/single_inference`: Generate a video from a text prompt
- `GET /api/video/status/{job_id}`: Poll for video generation status updates

## Video Status Updates

The status polling endpoint provides updates on the video generation process:

1. **pending**: Initial job status (queued)
2. **generating**: Video generation in progress (NVIDIA API processing)
3. **processing**: Processing video files (extracting from zip)
4. **complete**: Video generation complete (with video URL)
5. **failed**: Video generation failed (with error message)

## Frontend Examples

The `frontend_examples` directory contains the following ready-to-use HTML examples:

- `video_generator_complete.html`: **Recommended comprehensive example** that includes both single and batch video generation with polling
- `polling_example.html`: Minimal example showing the polling implementation pattern
- `batch_video_generator.html`: Legacy example retained for reference (using polling)

## CLI Simulator

For testing the prompt tuning functionality, you can use the CLI simulator:

```bash
# Interactive Mode
python cli_simulator.py interactive

# Enhance a rough prompt
python cli_simulator.py enhance "Three drones flying over earthquake mountains cabin"

# Initialize with a prompt
python cli_simulator.py initialize "At dawn, three matte-black quad-rotor drones hover over a landslide-stricken mountain slope."

# Update a prompt
python cli_simulator.py update "make it night time"

# Generate variations from selected prompts (by index)
python cli_simulator.py variations 0,2 8

# View prompt history
python cli_simulator.py history
```

## Deploying on Brev

This project is ready to be deployed on a Brev server instance:

1. Clone the repository on your Brev instance
2. Install dependencies with `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file with your API keys
4. Run the server with `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. The server will be available at the Brev instance's public IP/domain

## Project Structure

```
.
├── app/
│   ├── api/                 # API routes and schemas
│   ├── core/                # Core config and utilities 
│   ├── services/            # Business logic services
│   │   ├── batch_inference_service.py  # Multi-GPU batch inference
│   │   ├── parameter_extractor.py      # Extract parameters from prompts
│   │   ├── prompt_manager.py           # Manage prompt refinement
│   │   └── video_service.py            # Interface with Cosmos API
│   ├── utils/               # Helper utilities
│   └── main.py              # FastAPI application
├── batch_inference.py       # Standalone batch inference script
├── example_prompts.json     # Example prompts for batch inference
├── frontend_examples/       # HTML demo interfaces
├── static/                  # Static files directory
│   └── videos/              # Generated videos
├── requirements.txt         # Python dependencies
├── start_batch_server.sh    # Start server with batch inference
├── start_ngrok.sh           # Script to start ngrok
├── test_batch_inference.py  # Test script for batch inference
├── BATCH_INFERENCE.md       # Documentation for batch inference
└── README.md                # Project documentation
```

## Implementation Details

### Minimal Prompt Changes

The system implements minimal prompt changes through several key components:

1. **Parameter Extraction**: Precisely identifies parameters from the initial prompt
2. **Update Request Processing**: Analyzes user requests to determine exactly which parameters need to change
3. **Minimal Prompt Regeneration**: Instructs the language model to:
   - Focus only on changing components mentioned in the update request
   - Maintain the same structure and flow of the original prompt
   - Preserve the style and tone of the original prompt
   - Avoid adding unnecessary details not present in the original prompt

### Video Generation

The video generation process utilizes NVIDIA's Cosmos Text2World model:

1. User submits a prompt through the API
2. The system generates a unique job ID and starts the video generation asynchronously
3. Client connects to WebSocket endpoint to receive real-time status updates
4. Server calls NVIDIA Cosmos API and monitors the generation process
5. Generated video is extracted from the resulting ZIP file and made available
6. Client receives the video URL via WebSocket when generation is complete

### Batch Video Generation

For high-throughput scenarios, the system supports batch video generation:

1. Submit up to 8 prompts via the batch inference API endpoint
2. Each prompt is assigned to a different GPU for parallel execution
3. A batch ID is returned for tracking all jobs in the batch
4. Real-time status updates are provided via WebSocket
5. All generated videos are tracked in a single batch response

For more details, see [BATCH_INFERENCE.md](BATCH_INFERENCE.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.