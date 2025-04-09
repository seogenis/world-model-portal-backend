from typing import Dict, Any, List, Tuple, Set, Optional
import json
import re
import time
from openai import OpenAI
from pydantic import BaseModel, Field, validator
from app.core.config import get_settings
from app.core.logger import get_logger, log_parameters, log_parameter_changes, log_prompt_change
from app.services.parameter_extractor import ParameterExtractor

settings = get_settings()
logger = get_logger()

class PromptVariation(BaseModel):
    """Model for a single prompt variation."""
    text: str = Field(..., min_length=10)

class PromptVariationsResponse(BaseModel):
    """Model for prompt variations response validation."""
    variations: List[PromptVariation] = Field(..., min_items=1)
    
    @validator('variations')
    def variations_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("At least one prompt variation must be provided")
        return v


class PromptManager:
    """Manages prompt parameters and updates with session support."""
    
    def __init__(self, parameter_extractor: ParameterExtractor, session_id: Optional[str] = None):
        self.parameter_extractor = parameter_extractor
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.current_parameters: Dict[str, Any] = {}
        self.original_prompt: str = ""
        self.current_prompt: str = ""
        self.update_history: List[Dict[str, Any]] = []
        self.prompt_history: List[Dict[str, Any]] = []  # Stores all past prompts with their parameters
        self.session_id: Optional[str] = session_id
        self.last_accessed: float = time.time()  # Track when this manager was last used
    
    def __getstate__(self):
        """Return state for pickling, excluding the OpenAI client."""
        state = self.__dict__.copy()
        # Remove the OpenAI client which can't be pickled
        state.pop('client', None)
        # Remove parameter_extractor reference which shouldn't be pickled
        state.pop('parameter_extractor', None)
        return state
        
    def __setstate__(self, state):
        """Restore instance from pickled state."""
        self.__dict__.update(state)
        # Recreate the OpenAI client
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # parameter_extractor is reattached during unpickling in the session manager
    
    def to_json(self) -> str:
        """Serialize the PromptManager state to JSON string for storage."""
        # Ensure history doesn't grow too large (keep only the last 25 entries)
        MAX_HISTORY_ENTRIES = 25
        
        if len(self.update_history) > MAX_HISTORY_ENTRIES:
            self.update_history = self.update_history[-MAX_HISTORY_ENTRIES:]
            logger.info(f"Session {self.session_id}: Trimmed update history to {MAX_HISTORY_ENTRIES} entries")
            
        if len(self.prompt_history) > MAX_HISTORY_ENTRIES:
            self.prompt_history = self.prompt_history[-MAX_HISTORY_ENTRIES:]
            logger.info(f"Session {self.session_id}: Trimmed prompt history to {MAX_HISTORY_ENTRIES} entries")
        
        serializable_data = {
            "current_parameters": self.current_parameters,
            "original_prompt": self.original_prompt,
            "current_prompt": self.current_prompt, 
            "update_history": self.update_history,
            "prompt_history": self.prompt_history,
            "session_id": self.session_id,
            "last_accessed": self.last_accessed
        }
        return json.dumps(serializable_data)
        
    @classmethod
    def from_json(cls, json_str: str, parameter_extractor: ParameterExtractor) -> 'PromptManager':
        """Create a PromptManager instance from JSON string."""
        try:
            data = json.loads(json_str)
            
            # Create a new instance
            instance = cls(parameter_extractor)
            
            # Restore the state
            instance.current_parameters = data.get("current_parameters", {})
            instance.original_prompt = data.get("original_prompt", "")
            instance.current_prompt = data.get("current_prompt", "")
            instance.update_history = data.get("update_history", [])
            instance.prompt_history = data.get("prompt_history", [])
            instance.session_id = data.get("session_id")
            instance.last_accessed = data.get("last_accessed", time.time())
            
            return instance
            
        except Exception as e:
            logger.error(f"Error deserializing PromptManager: {str(e)}")
            # Return a fresh instance if deserialization fails
            return cls(parameter_extractor)
            
    def touch(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = time.time()
        
    async def initialize_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Initialize parameters from a new prompt."""
        logger.info(f"Session {self.session_id}: Initializing from prompt: '{prompt[:50]}...'")
        self.original_prompt = prompt
        self.current_prompt = prompt
        self.current_parameters = await self.parameter_extractor.extract_parameters(prompt)
        
        # Log the extracted parameters
        log_parameters(self.current_parameters, f"Session {self.session_id}: Extracted")
        
        # Reset history
        self.update_history = []
        
        # Add initial prompt to history
        self.prompt_history = [{
            "prompt": prompt,
            "parameters": self.current_parameters.copy(),
            "description": "Initial prompt"
        }]
        
        # Update last accessed time
        self.touch()
        
        return self.current_parameters
    
    async def process_update_request(self, user_request: str) -> Tuple[Dict[str, Any], List[str]]:
        """Process user request to update parameters."""
        logger.info(f"Session {self.session_id}: Processing update request: '{user_request}'")
        
        # Update last accessed time
        self.touch()
        
        # Create the tools parameter for function calling
        tools = [
            {
                "type": "function",
                "function": {
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
                }
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=settings.UPDATE_REQUEST_MODEL,
                messages=[
                    {"role": "system", "content": """
                    Identify which parameters the user wants to modify based on their natural language request.
                    
                    IMPORTANT GUIDELINES:
                    1. Only update parameters that are EXPLICITLY mentioned in the user request.
                    2. Be precise and minimal in your changes - don't modify parameters unless directly referenced.
                    3. Infer parameter names from context when the user provides descriptive changes.
                    4. For completely new settings or locations, update the relevant parameters accordingly.
                    5. If user mentions changing the atmosphere/mood, update the 'atmosphere' parameter.
                    6. If user mentions changing the setting/place, update the 'location' parameter.
                    7. ONLY include parameters that need to change in your response.
                    
                    Format your response as a clean JSON:
                    {
                      "updates": {
                        "parameter_name": "new_value",
                        ...
                      },
                      "changes_description": [
                        "Updated 'parameter_name' from 'old_value' to 'new_value'.",
                        ...
                      ]
                    }
                    """}, 
                    {"role": "user", "content": f"Current parameters: {json.dumps(self.current_parameters, indent=2)}\n\nCurrent prompt: {self.current_prompt}\n\nUser request: {user_request}"}
                ]
            )
            
            # Try to get function call result first, then fall back to content
            message = response.choices[0].message
            content = ""
            
            # Check if there's a function call response
            if hasattr(message, 'function_call') and message.function_call:
                # Extract from function call
                try:
                    content = message.function_call.arguments
                    logger.info(f"Update response from function call: {content[:100]}...")
                except AttributeError:
                    logger.warning("Function call present but couldn't access arguments")
            
            # Fall back to content if function call is not available or empty
            if not content:
                content = message.content or ""
                logger.info(f"Update response from content: {content[:100]}...")
            
            # Try to parse JSON from the response
            try:
                # Find JSON in the content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = content[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    if 'updates' in parsed_data and 'changes_description' in parsed_data:
                        updates = parsed_data['updates']
                        changes_description = parsed_data['changes_description']
                    else:
                        # Try to infer what the updates are based on common patterns
                        updates = {}
                        changes_description = []
                        
                        # Location changes
                        for location_term in ["beach", "forest", "city", "shore", "ocean", "mountain", "desert", "space"]:
                            if location_term in user_request.lower() and "location" in self.current_parameters:
                                old_location = self.current_parameters["location"]
                                updates["location"] = location_term
                                changes_description.append(f"Updated 'location' from '{old_location}' to '{location_term}'.")
                                break
                        
                        # Color changes
                        for color in ["blue", "red", "green", "yellow", "purple", "orange", "black", "white"]:
                            if color in user_request.lower() and "colors" in self.current_parameters:
                                old_color = self.current_parameters["colors"]
                                updates["colors"] = color
                                changes_description.append(f"Updated 'colors' from '{old_color}' to '{color}'.")
                                break
                        
                        # Time changes
                        for time in ["day", "night", "morning", "evening", "afternoon", "dusk", "dawn"]:
                            if time in user_request.lower() and "time_of_day" in self.current_parameters:
                                old_time = self.current_parameters["time_of_day"]
                                updates["time_of_day"] = time
                                changes_description.append(f"Updated 'time_of_day' from '{old_time}' to '{time}'.")
                                break
                        
                        # If we couldn't infer specific changes
                        if not updates:
                            changes_description = [f"Processed request: {user_request}"]
                else:
                    # Simple fallback
                    updates = {}
                    changes_description = [f"Processed: {user_request}"]
            except Exception as parse_error:
                logger.error(f"Error parsing update response: {str(parse_error)}")
                updates = {}
                changes_description = [f"Processed request: {user_request}"]
                
        except Exception as e:
            logger.error(f"Error processing update request: {str(e)}")
            # Return minimal updates to avoid breaking the flow
            return self.current_parameters, [f"Error processing request: {str(e)}"]
        
        # If no updates were detected, provide feedback
        if not updates:
            return self.current_parameters, ["No specific parameters were identified to update."]
        
        # Log the updates
        log_parameters(updates, "Updates")
        
        # Save a copy of the previous parameters for logging
        previous_parameters = self.current_parameters.copy()
        
        # Update parameters
        self.current_parameters.update(updates)
        
        # Log the actual changes that were made
        log_parameter_changes(previous_parameters, self.current_parameters)
        
        # Add to history
        self.update_history.append({
            "request": user_request,
            "updates": updates,
            "changes": changes_description
        })
        
        return self.current_parameters, changes_description
    
    async def regenerate_prompt(self) -> str:
        """Regenerate a coherent prompt from the current parameters."""
        logger.info(f"Session {self.session_id}: Regenerating prompt from current parameters")
        
        # Update last accessed time
        self.touch()
        
        try:
            # Get the most recent update history to understand what changed
            recent_changes = []
            change_description = "Updated prompt"
            if self.update_history:
                recent_changes = self.update_history[-1]["changes"]
                change_description = self.update_history[-1]["request"]
            
            response = self.client.chat.completions.create(
                model=settings.PROMPT_REGENERATION_MODEL,
                messages=[
                    {"role": "system", "content": """
                    Generate a coherent, detailed text-to-video prompt using the provided parameters.
                    The prompt should flow naturally and include all the specified parameters.
                    
                    IMPORTANT INSTRUCTIONS:
                    1. Maintain the style and tone of the original prompt.
                    2. Make MINIMAL changes to the previous prompt - only change what is necessary.
                    3. Focus on changing ONLY the components mentioned in the recent changes.
                    4. Keep the same sentence structure and flow when possible.
                    5. Do not add unnecessary details or embellishments unless they were in the previous prompt.
                    6. Ensure the new prompt fully incorporates all current parameters.
                    """}, 
                    {"role": "user", "content": f"Original prompt: {self.original_prompt}\n\nPrevious prompt: {self.current_prompt}\n\nCurrent parameters: {json.dumps(self.current_parameters, indent=2)}\n\nRecent changes: {json.dumps(recent_changes, indent=2)}\n\nGenerate a new coherent prompt that maintains as much of the previous prompt as possible while incorporating ONLY the necessary changes."}
                ]
            )
            
            new_prompt = response.choices[0].message.content
            
            # Log the difference between the old and new prompts
            previous_prompt = self.current_prompt
            log_prompt_change(previous_prompt, new_prompt)
            
            # Update the current prompt
            self.current_prompt = new_prompt
            
            # Add to prompt history
            self.prompt_history.append({
                "prompt": new_prompt,
                "parameters": self.current_parameters.copy(),
                "description": change_description
            })
            
            logger.info(f"Session {self.session_id}: Generated new prompt: '{new_prompt[:50]}...'")
            return new_prompt
            
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error regenerating prompt: {str(e)}")
            # Return the existing prompt if regeneration fails
            return self.current_prompt or self.original_prompt
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of updates made to the prompt."""
        return self.update_history
        
    def get_prompt_history(self) -> List[Dict[str, Any]]:
        """Get the history of all prompts."""
        return self.prompt_history
        
    async def enhance_prompt(self, rough_prompt: str) -> str:
        """Enhance a rough prompt with descriptive details.
        
        Prompts over 50 words are returned as-is. Prompts under 50 words are enhanced
        with descriptive details about environment, lighting, atmosphere, etc.
        """
        logger.info(f"Session {self.session_id}: Enhancing rough prompt: '{rough_prompt[:50]}...'")
        
        # Update last accessed time
        self.touch()
        
        # If the prompt is over 50 words, return it as is; otherwise enhance it
        word_count = len(rough_prompt.split())
        if word_count > 50:
            logger.info(f"Session {self.session_id}: Prompt is already detailed ({word_count} words), returning as is")
            return rough_prompt
        
        try:
            response = self.client.chat.completions.create(
                model=settings.PROMPT_ENHANCEMENT_MODEL,
                messages=[
                    {"role": "system", "content": f"""
                    Enhance a rough text-to-video prompt with descriptive visual details.
                    
                    You are ONLY enhancing prompts that are under 50 words. The system has already filtered prompts longer than 50 words.
                    
                    IMPORTANT GUIDELINES:
                    1. DO NOT introduce new subjects or actions that were not in the original prompt.
                    2. DO elaborate on visual details of subjects, environment, lighting, atmosphere, etc.
                    3. Use rich, descriptive language to paint a vivid visual scene.
                    4. Maintain the core meaning and intent of the original prompt.
                    5. Focus on enhancing ENVIRONMENTAL details, visual characteristics, and atmosphere.
                    6. Include specific details about lighting, textures, colors, and spatial relationships.
                    7. Create a prompt that would help a text-to-video model generate a high-quality, detailed scene.
                    8. Keep your response CONCISE and NOT VERBOSE.
                    9. For very short prompts (under 20 words), you can be more detailed.
                    10. For prompts between 20-50 words, be more moderate with your additions.
                    
                    EXAMPLE:
                    User: "3 drones flying over mountain with earthquake at dawn"
                    Enhanced: "Three matte-black quad-rotor drones hover over a rugged mountain slope at dawn, their red and green LEDs pulsing through low rolling mist. Sharp rays of pale orange sunlight reveal fallen trees, scattered debris, and the remains of a rural settlement—a collapsed cabin and bent metal sheets—left by a recent landslide. Photorealistic"
                    """}, 
                    {"role": "user", "content": f"Rough prompt: {rough_prompt}\n\nEnhance this prompt with rich visual details while preserving all original subjects and actions. Focus on making the environment, lighting, atmosphere, and visual characteristics more descriptive. The user prompt is {word_count} words - keep your enhancement appropriately sized."}
                ]
            )
            
            enhanced_prompt = response.choices[0].message.content
            # Strip single and double quotation marks from the response
            enhanced_prompt = enhanced_prompt.replace('"', '').replace("'", '')
            logger.info(f"Enhanced prompt: '{enhanced_prompt[:50]}...'")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            # Return the original prompt if enhancement fails
            return rough_prompt
            
    async def generate_prompt_variations(self, selected_prompt_indices: List[int], total_count: int = 8) -> List[str]:
        """Generate variations of selected prompts to have a total of 'total_count' prompts."""
        logger.info(f"Generating variations based on {len(selected_prompt_indices)} selected prompts to reach {total_count} total")
        
        if not self.prompt_history:
            logger.error("No prompts in history to generate variations from")
            return []
            
        # Validate indices
        valid_indices = [i for i in selected_prompt_indices if 0 <= i < len(self.prompt_history)]
        if not valid_indices:
            logger.error(f"No valid prompt indices provided: {selected_prompt_indices}")
            return []
            
        # Get selected prompts
        selected_prompts = [self.prompt_history[i]["prompt"] for i in valid_indices]
        selected_parameters = [self.prompt_history[i]["parameters"] for i in valid_indices]
        
        # Always generate (total_count - selected_count) new prompts to ensure we reach exactly total_count
        num_to_generate = max(1, total_count - len(selected_prompts))
        
        # Ensure we're returning exactly total_count prompts (default 8) after combining
        # If we have more selected prompts than total_count, trim the selected prompts
        if len(selected_prompts) >= total_count:
            logger.info(f"Selected prompts ({len(selected_prompts)}) exceed requested total ({total_count}), returning first {total_count}")
            return selected_prompts[:total_count]
            
        try:
            # Prepare a system prompt based on the number of selected prompts
            if len(selected_prompts) == 1:
                system_prompt = """
                Generate variations of the provided prompt by modifying environmental details while keeping the main subjects the same.
                
                IMPORTANT GUIDELINES:
                1. Preserve the main subjects and their key attributes in all variations.
                2. Vary the environment, setting, time of day, lighting, or atmosphere in each variation.
                3. Ensure each variation maintains the same quality and level of detail as the original.
                4. Make each variation meaningfully different from the others.
                5. Keep the style consistent across all variations.
                """
            else:
                system_prompt = """
                Generate new prompt variations based on the patterns observed in the provided prompts.
                
                IMPORTANT GUIDELINES:
                1. Identify the similarities and differences between the provided prompts.
                2. Generate variations that follow the same pattern of changes.
                3. If the prompts differ in setting (forest vs. mountains), create variations with new settings.
                4. If the prompts differ in time (day vs. night), create variations with different times.
                5. If the prompts differ in subject actions, create variations with new logical actions.
                6. Maintain the same level of detail and style as the original prompts.
                7. Ensure variations are meaningful and consistent with the theme of the originals.
                """
                
            # Update the prompt to explicitly request numbered variations with clear formatting
            response = self.client.chat.completions.create(
                model=settings.PROMPT_VARIATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt + """
                    IMPORTANT FORMAT INSTRUCTIONS:
                    1. ALWAYS return EXACTLY the requested number of variations.
                    2. Format each variation on a new line as: "Variation N: [prompt text]"
                    3. DO NOT add any explanations, comments, or additional text.
                    4. Each variation should be a complete, standalone prompt.
                    5. Ensure CLEAR SEPARATION between each variation with consistent formatting.
                    """},
                    {"role": "user", "content": f"Original prompts:\n\n" + "\n\n".join([f"Prompt {i+1}: {prompt}" for i, prompt in enumerate(selected_prompts)]) + 
                    f"\n\nParameters for reference:\n\n" + "\n\n".join([f"Parameters {i+1}: {json.dumps(params, indent=2)}" for i, params in enumerate(selected_parameters)]) +
                    f"\n\nI need EXACTLY {num_to_generate} new prompt variations. You MUST return precisely {num_to_generate} variations, each clearly marked with 'Variation N:' at the start of the line."}
                ]
            )
            
            variations_text = response.choices[0].message.content
            logger.debug(f"Raw variations response: {variations_text}")
            
            # Use a more robust regex approach to extract variations
            variation_pattern = re.compile(r'(?:Prompt|Variation)[^\n:]*:[\s]*(.*?)(?=\n(?:Prompt|Variation)[^\n:]*:|\Z)', re.DOTALL)
            matches = list(variation_pattern.finditer(variations_text))
            
            # Extract the variations from the regex matches
            variations = []
            for match in matches:
                variation_text = match.group(1).strip()
                if variation_text:  # Only add non-empty variations
                    variations.append(variation_text)
            
            # If we still don't have enough variations, try a more aggressive approach
            if len(variations) < num_to_generate:
                logger.warning(f"First extraction method only found {len(variations)} variations, trying alternate method")
                
                # Alternate method 1: Split by numbered markers (1., 2., etc.)
                numbered_pattern = re.compile(r'(?:^|\n)\s*\d+\.\s*(.*?)(?=(?:\n\s*\d+\.)|$)', re.DOTALL)
                numbered_matches = list(numbered_pattern.finditer(variations_text))
                
                if numbered_matches and len(numbered_matches) >= num_to_generate:
                    variations = [match.group(1).strip() for match in numbered_matches if match.group(1).strip()]
                else:
                    # Alternate method 2: Just split by double newlines
                    logger.warning("Trying to split by double newlines")
                    paragraphs = [p.strip() for p in variations_text.split('\n\n') if p.strip()]
                    if paragraphs and len(paragraphs) >= num_to_generate:
                        variations = paragraphs
                    else:
                        # Last resort: Split by single newlines
                        logger.warning("Trying to split by single newlines")
                        lines = [line.strip() for line in variations_text.split('\n') if line.strip()]
                        
                        # Filter lines that are likely actual prompts (longer than 20 chars and not starting with special characters)
                        variations = [line for line in lines if len(line) > 20 and not line.startswith(('*', '-', '#'))]
            
            logger.info(f"Generated {len(variations)} prompt variations")
            
            # Combine original selected prompts with new variations
            all_prompts = selected_prompts + variations
            
            # Ensure we have exactly total_count prompts
            if len(all_prompts) < total_count:
                # If we don't have enough variations, duplicate the last few variations to reach total_count
                logger.warning(f"Only generated {len(all_prompts)} total prompts, need {total_count}. Duplicating to reach target.")
                
                # Duplicate variations in round-robin fashion until we have enough
                i = 0
                while len(all_prompts) < total_count and i < 100:  # Prevent infinite loop
                    # Try to add from variations first, then from selected prompts if needed
                    source_prompts = variations if variations else selected_prompts
                    duplicate_idx = i % len(source_prompts)
                    prompt_to_duplicate = source_prompts[duplicate_idx]
                    all_prompts.append(f"{prompt_to_duplicate} (variation {i+1})")
                    i += 1
            
            # Ensure all prompts are valid using Pydantic validation
            valid_prompts = []
            for prompt in all_prompts[:total_count]:
                try:
                    # Validate each prompt meets minimum requirements
                    validated_prompt = PromptVariation(text=prompt)
                    valid_prompts.append(validated_prompt.text)
                except Exception as validation_error:
                    logger.warning(f"Prompt validation error: {validation_error}. Prompt: {prompt[:50]}...")
                    # If validation fails, substitute with a valid prompt from selected_prompts
                    if selected_prompts:
                        substitute_idx = len(valid_prompts) % len(selected_prompts)
                        valid_prompts.append(f"{selected_prompts[substitute_idx]} (substitute)")
            
            # Final validation as a collection to ensure we have exactly the right number
            try:
                # Ensure we have exactly total_count valid prompts
                final_valid_prompts = valid_prompts[:total_count]
                while len(final_valid_prompts) < total_count and selected_prompts:
                    idx = len(final_valid_prompts) % len(selected_prompts)
                    final_valid_prompts.append(f"{selected_prompts[idx]} (additional)")
                
                # Validate the entire collection
                validated_response = PromptVariationsResponse(
                    variations=[PromptVariation(text=p) for p in final_valid_prompts]
                )
                
                # Extract the validated prompt texts
                result = [v.text for v in validated_response.variations]
                logger.info(f"Returning exactly {len(result)} validated prompts")
                return result
                
            except Exception as collection_error:
                logger.error(f"Final validation error: {collection_error}")
                # Last resort fallback - just use selected prompts
                result = []
                i = 0
                while len(result) < total_count:
                    if selected_prompts:
                        idx = i % len(selected_prompts)
                        result.append(selected_prompts[idx])
                    else:
                        result.append(f"A detailed scene with interesting lighting and atmosphere (fallback {i+1})")
                    i += 1
                return result[:total_count]
            
        except Exception as e:
            logger.error(f"Error generating prompt variations: {str(e)}")
            
            # Even on error, make sure we return total_count prompts
            try:
                result = []
                
                # Use selected prompts first
                if selected_prompts:
                    result.extend(selected_prompts)
                
                # If we need more, create fallbacks
                while len(result) < total_count:
                    if selected_prompts:
                        # Duplicate from selected prompts
                        idx = len(result) % len(selected_prompts)
                        result.append(f"{selected_prompts[idx]} (fallback variation)")
                    else:
                        # If no selected prompts, create generic ones
                        result.append(f"A detailed cinematic scene with atmospheric lighting (fallback {len(result)+1})")
                
                # Validate with Pydantic
                validated_result = []
                for prompt in result[:total_count]:
                    try:
                        validated = PromptVariation(text=prompt)
                        validated_result.append(validated.text)
                    except:
                        validated_result.append(f"A detailed scene with interesting visuals (emergency fallback {len(validated_result)+1})")
                
                logger.info(f"Returning {len(validated_result)} fallback prompts after error")
                return validated_result
                
            except Exception as fallback_error:
                # Ultra fallback - just return generic prompts
                logger.error(f"Critical fallback error: {fallback_error}")
                return [f"A detailed scene with interesting visuals (emergency fallback {i+1})" for i in range(total_count)]