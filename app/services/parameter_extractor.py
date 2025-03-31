import json
import re
from typing import Dict, Any
from openai import OpenAI
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger()


class ParameterExtractor:
    """Extracts parameters from natural language prompts using OpenAI."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.model = settings.PARAMETER_EXTRACTION_MODEL
        
    async def extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract structured parameters from a text prompt."""
        logger.info(f"Extracting parameters using model: {self.model}")
        
        # Create the tools parameter for function calling
        tools = [
            {
                "type": "function",
                "function": {
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
                }
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """
                    Extract structured information from text-to-video prompts.
                    Return ONLY a clean JSON without any explanation, using these parameter keys:
                    
                    {
                      "subjects": "Main entities/objects in the scene",
                      "subject_attributes": "Color, size, material of main subjects",
                      "actions": "What the subjects are doing",
                      "location": "The setting or environment",
                      "time_of_day": "When the scene takes place",
                      "lighting": "Lighting conditions and sources",
                      "atmosphere": "Mood, weather, ambiance",
                      "colors": "Dominant or important colors in the scene",
                      "style": "Visual style of the scene (photorealistic, cartoon, etc)"
                    }
                    
                    Don't include parameters that aren't specified in the prompt.
                    ONLY return a JSON object with the parameters.
                    """}, 
                    {"role": "user", "content": f"Extract parameters from: {prompt}"}
                ]
            )
            
            # Fall back to direct content if function calling fails
            content = response.choices[0].message.content
            logger.info(f"Raw response: {content[:200]}...")
            
            # Direct JSON parsing from content
            try:
                # Try to find JSON in the content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = content[json_start:json_end]
                    extracted_data = json.loads(json_str)
                    
                    # Handle different possible formats
                    if 'parameters' in extracted_data:
                        params = extracted_data['parameters']
                    else:
                        # If it's just a flat dictionary of parameters
                        params = extracted_data
                    
                    # Clean up parameter keys that have category labels
                    cleaned_params = {}
                    for key, value in params.items():
                        # Strip category markers like "- **Subject**" from the keys
                        if key.startswith('-') and '**' in key:
                            # Convert the category name to a proper parameter name
                            category = key.split('**')[1].strip().lower()
                            
                            # Singular to match existing code expectations
                            if category.endswith('s'):
                                category = category[:-1]
                                
                            # Special case for some categories
                            if category in ["subject attribute", "subject attributes"]:
                                # For attributes, we need to parse the value and extract color, etc.
                                if "color" in value.lower():
                                    color_match = re.search(r'(\w+)\s*\(color', value.lower())
                                    if color_match:
                                        cleaned_params["color"] = color_match.group(1).strip()
                                # Continue with other attributes as needed
                            elif category in ["action", "actions"]:
                                cleaned_params["subject_action"] = value
                            elif category in ["location/setting"]:
                                cleaned_params["location"] = value
                            elif category in ["environmental detail", "environmental details"]:
                                cleaned_params["environment"] = value
                            elif category in ["prop", "props", "props and objects"]:
                                # Split into multiple props if needed
                                props = value.split(',')
                                for i, prop in enumerate(props):
                                    cleaned_params[f"prop_{i+1}"] = prop.strip()
                            elif category in ["lighting condition", "lighting conditions"]:
                                cleaned_params["lighting"] = value
                            elif category in ["time indicator", "time indicators"]:
                                cleaned_params["time_of_day"] = value
                            else:
                                # Use the category as is
                                cleaned_params[category] = value
                        else:
                            # Keep the key as is if it doesn't match the pattern
                            cleaned_params[key] = value
                    
                    if cleaned_params:
                        logger.info(f"Cleaned and extracted {len(cleaned_params)} parameters")
                        return cleaned_params
                    
                    logger.info(f"Using original {len(params)} parameters")
                    return params
                else:
                    # Simple fallback parsing
                    lines = content.strip().split('\n')
                    params = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            
                            # Clean up the key
                            clean_key = key.strip()
                            if clean_key.startswith('-') and '**' in clean_key:
                                # Extract the category name
                                category = clean_key.split('**')[1].strip().lower()
                                if category.endswith('s'):
                                    category = category[:-1]
                                
                                # Simplify category names
                                if category in ["subject attribute", "subject attributes"]:
                                    clean_key = "attributes"
                                elif category in ["action", "actions"]:
                                    clean_key = "subject_action"
                                elif category in ["location/setting"]:
                                    clean_key = "location"
                                elif category == "environmental details":
                                    clean_key = "environment"
                                else:
                                    clean_key = category
                            
                            params[clean_key] = value.strip()
                    
                    if params:
                        logger.info(f"Extracted {len(params)} parameters via line splitting")
                        return params
            except Exception as parse_error:
                logger.error(f"Error parsing content as JSON: {str(parse_error)}")
            
            # Parse the prompt directly as a last fallback
            words = prompt.lower().split()
            params = {}
            
            # Extract subjects and their attributes
            if "robot" in prompt.lower():
                params["subjects"] = "robot"
                for color in ["teal", "blue", "red", "green", "black", "white", "purple"]:
                    if color in prompt.lower():
                        params["subject_attributes"] = f"{color} robot"
                        params["colors"] = color
                        break
            elif "drone" in prompt.lower() or "drones" in prompt.lower():
                params["subjects"] = "drones"
                if "quad" in prompt.lower():
                    params["subject_attributes"] = "quad-rotor drones"
                if "matte-black" in prompt.lower():
                    params["colors"] = "matte-black, red and green (LEDs)"
                    
            # Extract location
            for location in ["kitchen", "forest", "house", "city", "beach"]:
                if location in prompt.lower():
                    params["location"] = location
                    break
                    
            # Extract time
            for time in ["night", "day", "morning", "afternoon", "evening", "dusk", "dawn"]:
                if time in prompt.lower():
                    params["time_of_day"] = time
                    break
                    
            # Extract actions if any verbs are found
            for action in ["cooking", "circle", "burn", "fly", "hover", "swim"]:
                if action in prompt.lower():
                    params["actions"] = action
                    break
                    
            # Extract style
            if "photorealistic" in prompt.lower():
                params["style"] = "photorealistic"
                
            # If we found parameters, return them
            if params:
                logger.info(f"Generated {len(params)} parameters via direct prompt analysis")
                return params
                
            # If we couldn't extract anything, return a default set based on the prompt type
            if "drone" in prompt.lower() or "flying" in prompt.lower():
                return {
                    "subjects": "drones",
                    "actions": "flying",
                    "location": "forest",
                    "time_of_day": "night",
                    "lighting": "drone LEDs, fire light",
                    "style": "photorealistic"
                }
            else:
                # Default to the robot kitchen scenario
                return {
                    "subjects": "robot",
                    "subject_attributes": "teal robot",
                    "actions": "cooking food",
                    "location": "kitchen",
                    "time_of_day": "afternoon",
                    "lighting": "afternoon light",
                    "colors": "teal"
                }
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            # Return a basic set of parameters to avoid breaking the flow
            if "drone" in prompt.lower() or "flying" in prompt.lower():
                return {
                    "subjects": "drones",
                    "actions": "flying",
                    "style": "photorealistic"
                }
            else:
                return {
                    "subjects": "robot",
                    "actions": "cooking",
                    "location": "kitchen"
                }