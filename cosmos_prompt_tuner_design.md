# Cosmos Prompt Tuning Agent - Design Document

## Overview

This document outlines the design for a prompt-tuning agent for NVIDIA's Cosmos text-to-video model. The agent will help users refine and modify text prompts for video generation through intuitive interactions, modularizing prompts into parameters that can be independently modified.

## Goals

1. Extract structured parameters from natural language prompts
2. Allow users to modify specific aspects of the prompt through conversation
3. Maintain prompt coherence while changing parameters
4. Minimize latency in the interaction loop
5. Support diverse prompt structures without hard-coding parameters

## System Architecture

### Component Overview

```
agent_prompt_tuner/
├── app/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Pydantic models for API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   └── logger.py         # Logging configuration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parameter_extractor.py  # OpenAI-based parameter extraction
│   │   ├── prompt_manager.py       # Parameter tracking and updates
│   │   └── llm_service.py          # LLM API interface
│   └── utils/
│       ├── __init__.py
│       └── openai_utils.py   # OpenAI API helper functions
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   ├── test_parameter_extractor.py
│   ├── test_prompt_manager.py
│   └── test_api.py
├── cli_simulator.py          # Frontend simulation for testing
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## User Flow

### Example User Flow

1. **Initial Prompt Submission**:
   - User submits: "The teal robot is cooking food in a kitchen. Steam rises from a simmering pot as the robot chops vegetables on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered measuring spoons and a half-empty bottle of olive oil."
   - System extracts parameters using o1-mini (optimized for speed)
   - Parameters extracted:
     ```json
     {
       "subject": "teal robot",
       "subject_action": "cooking food, chopping vegetables",
       "location": "kitchen",
       "time_of_day": "afternoon",
       "prop_1": "simmering pot with steam",
       "prop_2": "worn wooden cutting board",
       "prop_3": "copper pans on overhead rack",
       "prop_4": "cast iron skillet",
       "prop_5": "measuring spoons",
       "prop_6": "half-empty bottle of olive oil",
       "lighting": "afternoon light, glints on copper pans",
       "atmosphere": "well-used cooking space"
     }
     ```
   - Parameters and original prompt displayed to user

2. **Parameter Modification Request**:
   - User says: "Change the robot to blue and make it baking a cake instead"
   - System processes with 4o-mini to identify affected parameters:
     ```json
     {
       "updates": {
         "subject": "blue robot",
         "subject_action": "baking a cake"
       },
       "changes_description": [
         "Changed robot color from teal to blue",
         "Changed activity from cooking food/chopping vegetables to baking a cake"
       ]
     }
     ```

3. **Prompt Regeneration**:
   - System uses 4o-mini to generate a coherent new prompt:
   - "The blue robot is baking a cake in a kitchen. Steam rises from a simmering pot as the robot measures ingredients on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered measuring spoons and a half-empty bottle of olive oil."
   - Updated parameters and prompt displayed to user

4. **Further Refinement**:
   - User says: "Remove the skillet and add a stand mixer"
   - System processes with 4o-mini:
     ```json
     {
       "updates": {
         "prop_4": "stand mixer",
       },
       "changes_description": [
         "Replaced cast iron skillet with stand mixer"
       ]
     }
     ```
   - Updated prompt generated
   - Iteration continues as needed

5. **Video Generation** (future integration):
   - User decides prompt is ready
   - System sends final prompt to Cosmos model for video generation
   - Video is generated and displayed to user

## Core Components and Functions

### 1. Parameter Extractor Service

**Purpose**: Extracts structured parameters from natural language prompts.

**Key Functions**:
- `extract_parameters(prompt: str) -> Dict[str, Any]`
  - Uses o1-mini for efficient parameter extraction
  - Implements function calling to enforce structured output
  - Returns dynamic set of parameters based on prompt content

**Example Implementation**:
```python
# app/services/parameter_extractor.py
import json
from typing import Dict, Any
import openai
from app.core.config import settings

class ParameterExtractor:
    """Extracts parameters from natural language prompts using OpenAI."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        self.client = openai.OpenAI(api_key=api_key)
        
    async def extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract structured parameters from a text prompt using o1-mini for speed."""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using o1-mini for faster extraction
            messages=[
                {"role": "system", "content": """
                Extract structured information from text-to-video prompts.
                Identify:
                - Subjects (entities, objects)
                - Subject attributes (color, size, material)
                - Actions 
                - Location/setting
                - Environmental details
                - Props and objects
                - Lighting conditions
                - Time indicators
                
                Create parameters dynamically based on what's in the prompt.
                Don't include parameters that aren't specified.
                """}, 
                {"role": "user", "content": f"Extract parameters from: {prompt}"}
            ],
            functions=[{
                "name": "extract_prompt_parameters",
                "description": "Extracts structured parameters from a text-to-video prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "parameters": {
                            "type": "object",
                            "description": "Extracted parameters. Keys are parameter names, values are extracted values.",
                            "additionalProperties": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["parameters"]
                }
            }],
            function_call={"name": "extract_prompt_parameters"}
        )
        
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        return function_args["parameters"]
```

### 2. Prompt Manager Service

**Purpose**: Tracks parameters and processes user modification requests.

**Key Functions**:
- `initialize_from_prompt(prompt: str) -> Dict[str, Any]`
  - Sets up initial parameters using the extractor
  - Stores original prompt and extracted parameters

- `process_update_request(user_request: str) -> Tuple[Dict[str, Any], List[str]]`
  - Uses 4o-mini to understand what user wants to change
  - Returns updated parameters and human-readable change descriptions
  
- `regenerate_prompt() -> str`
  - Creates a coherent prompt from modified parameters
  - Ensures natural flow between parameter elements

**Example Implementation**:
```python
# app/services/prompt_manager.py
from typing import Dict, Any, List, Tuple
import json
import openai
from app.core.config import settings
from app.services.parameter_extractor import ParameterExtractor

class PromptManager:
    """Manages prompt parameters and updates."""
    
    def __init__(self, parameter_extractor: ParameterExtractor):
        self.parameter_extractor = parameter_extractor
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.current_parameters: Dict[str, Any] = {}
        self.original_prompt: str = ""
        
    async def initialize_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Initialize parameters from a new prompt using faster o1-mini model."""
        self.original_prompt = prompt
        self.current_parameters = await self.parameter_extractor.extract_parameters(prompt)
        return self.current_parameters
    
    async def process_update_request(self, user_request: str) -> Tuple[Dict[str, Any], List[str]]:
        """Process user request to update parameters using 4o-mini for better comprehension."""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using 4o-mini for better understanding of user intent
            messages=[
                {"role": "system", "content": """
                Identify which parameters the user wants to modify and how.
                Return only the parameters that should be updated.
                Be precise in understanding subtle changes to the prompt.
                """}, 
                {"role": "user", "content": f"Current parameters: {self.current_parameters}\n\nUser request: {user_request}"}
            ],
            functions=[{
                "name": "identify_parameter_updates",
                "description": "Identifies which parameters should be updated based on user request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "object",
                            "description": "Parameters to update. Keys are parameter names, values are new values.",
                            "additionalProperties": {
                                "type": "string"
                            }
                        },
                        "changes_description": {
                            "type": "array",
                            "description": "Human-readable descriptions of the changes being made",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["updates", "changes_description"]
                }
            }],
            function_call={"name": "identify_parameter_updates"}
        )
        
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        updates = function_args["updates"]
        changes_description = function_args["changes_description"]
        
        # Update parameters
        self.current_parameters.update(updates)
        
        return self.current_parameters, changes_description
    
    async def regenerate_prompt(self) -> str:
        """Regenerate a coherent prompt from the current parameters."""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Good balance of quality and speed for regeneration
            messages=[
                {"role": "system", "content": """
                Generate a coherent, detailed text-to-video prompt using the provided parameters.
                The prompt should flow naturally and include all the specified parameters.
                Maintain the style and tone of the original prompt when possible.
                """}, 
                {"role": "user", "content": f"Original prompt: {self.original_prompt}\n\nCurrent parameters: {self.current_parameters}\n\nGenerate a new coherent prompt incorporating all parameters."}
            ]
        )
        
        return response.choices[0].message.content
```

### 3. API Interface

**Purpose**: Exposes prompt tuning functionality via FastAPI endpoints.

**Key Endpoints**:
- `POST /initialize`: Accept initial prompt and return extracted parameters
- `POST /update`: Process user modification request and return updated prompt
- `GET /parameters`: Get current parameters
- `GET /history`: Get history of prompt modifications

**Example Implementation**:
```python
# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

# Dependency injection
def get_parameter_extractor():
    return ParameterExtractor(settings.OPENAI_API_KEY)

def get_prompt_manager(parameter_extractor: ParameterExtractor = Depends(get_parameter_extractor)):
    return PromptManager(parameter_extractor)

# Pydantic models
class InitializeRequest(BaseModel):
    prompt: str

class UpdateRequest(BaseModel):
    user_request: str

class PromptResponse(BaseModel):
    parameters: Dict[str, Any]
    prompt: str
    changes: List[str] = []

@router.post("/initialize", response_model=PromptResponse)
async def initialize_prompt(
    request: InitializeRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Initialize the system with a new prompt."""
    try:
        parameters = await prompt_manager.initialize_from_prompt(request.prompt)
        return PromptResponse(
            parameters=parameters,
            prompt=request.prompt,
            changes=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing prompt: {str(e)}")

@router.post("/update", response_model=PromptResponse)
async def update_prompt(
    request: UpdateRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """Update parameters based on user request."""
    try:
        parameters, changes = await prompt_manager.process_update_request(request.user_request)
        new_prompt = await prompt_manager.regenerate_prompt()
        return PromptResponse(
            parameters=parameters,
            prompt=new_prompt,
            changes=changes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompt: {str(e)}")
```

### 4. CLI Simulator for Testing

**Purpose**: Simulates frontend interactions via command line.

**Key Functions**:
- Initialize with a prompt
- Process update requests
- Display changes and parameters
- Show regenerated prompts

**Example Implementation**:
```python
# cli_simulator.py
import asyncio
import json
import httpx
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

async def make_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()

async def main():
    print("=== Cosmos Prompt Tuner CLI Simulator ===")
    print("Enter an initial prompt to begin:")
    initial_prompt = input("> ")
    
    # Initialize with the prompt
    try:
        print("Processing initial prompt...")
        result = await make_request("/initialize", {"prompt": initial_prompt})
        
        print("\n=== Extracted Parameters ===")
        for param, value in result["parameters"].items():
            print(f"{param}: {value}")
        
        print("\n=== Current Prompt ===")
        print(result["prompt"])
        
        # Interactive loop
        while True:
            print("\nEnter modification request (or 'quit' to exit):")
            user_request = input("> ")
            
            if user_request.lower() == "quit":
                break
                
            print("Processing your request...")
            result = await make_request("/update", {"user_request": user_request})
            
            print("\n=== Changes Made ===")
            for change in result["changes"]:
                print(f"• {change}")
                
            print("\n=== Updated Parameters ===")
            for param, value in result["parameters"].items():
                print(f"{param}: {value}")
            
            print("\n=== Updated Prompt ===")
            print(result["prompt"])
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Optimization Strategies

### 1. Model Selection Based on Task

- Use **o1-mini** for initial parameter extraction (fast, good structure recognition)
- Use **4o-mini** for understanding user modifications (better comprehension with reasonable speed)
- Use **4o-mini** for prompt regeneration (good balance of quality and speed)

### 2. Caching Mechanism

```python
# app/services/cache_service.py
import json
import hashlib
from typing import Dict, Any, Optional
import redis
from app.core.config import settings

class CacheService:
    """Caching service for reducing API call latency."""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        self.ttl = settings.CACHE_TTL  # Time to live in seconds
        
    async def get_parameters(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get parameters from cache if they exist."""
        key = self._generate_key("parameters", prompt)
        cached_data = self.redis.get(key)
        if cached_data:
            return json.loads(cached_data)
        return None
    
    async def set_parameters(self, prompt: str, parameters: Dict[str, Any]) -> None:
        """Store parameters in cache."""
        key = self._generate_key("parameters", prompt)
        self.redis.setex(key, self.ttl, json.dumps(parameters))
        
    # Similar methods for caching update requests and regenerated prompts
```

### 3. Batch Processing for Multiple Parameters

When multiple parameters need updating in a single request, process them efficiently:

```python
# Efficient batch processing example
async def batch_update_parameters(self, updates: Dict[str, str]) -> None:
    """Update multiple parameters efficiently."""
    # Update in memory without additional API calls
    self.current_parameters.update(updates)
    
    # Only regenerate prompt once after all updates
    await self.regenerate_prompt()
```

### 4. Background Processing for Long-Running Tasks

Use background tasks for operations that might take longer:

```python
# app/api/routes.py
from fastapi import BackgroundTasks

@router.post("/initialize")
async def initialize_prompt(
    request: InitializeRequest,
    background_tasks: BackgroundTasks,
    prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    # Initial fast extraction
    parameters = await prompt_manager.initialize_from_prompt(request.prompt)
    
    # Schedule deeper parameter analysis in background
    background_tasks.add_task(prompt_manager.enhance_parameters, request.prompt)
    
    return PromptResponse(
        parameters=parameters,
        prompt=request.prompt,
        changes=[]
    )
```

## Future Enhancements

1. **Parameter Visualization**: Visual representation of parameters and relationships
2. **Prompt History**: Track changes and allow reverting to previous versions
3. **Parameter Suggestions**: Suggest additional parameters that might enhance the prompt
4. **Direct Cosmos Integration**: Send prompts directly to Cosmos for video generation
5. **User Preference Learning**: Adapt to user's preferred prompt style over time

## Implementation Timeline

1. **Week 1**: Core parameter extraction and prompt management
2. **Week 2**: API endpoints and CLI simulator
3. **Week 3**: Optimization and performance tuning
4. **Week 4**: Testing and refinement
5. **Week 5+**: Frontend integration and Cosmos model connection

## Conclusion

This prompt tuning agent provides an intuitive interface for modifying text-to-video prompts by extracting and manipulating key parameters. By using different OpenAI models optimized for specific tasks (o1-mini for fast extraction, 4o-mini for comprehension), the system balances performance and accuracy while maintaining a responsive user experience.

The architecture's flexibility allows it to handle diverse prompt styles without hard-coding parameters, making it adaptable to the wide range of creative inputs users might provide when generating videos with the Cosmos model.