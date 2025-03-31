from typing import Dict, Any, List, Optional
import json
from openai import OpenAI, APIError
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger()


class OpenAIClient:
    """Wrapper for OpenAI API client with error handling and retries."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
    
    async def chat_completion(
        self, 
        model: str,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request with error handling and retries."""
        
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                logger.debug(f"Sending request to {model} (attempt {attempt + 1}/{self.max_retries})")
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if functions is not None:
                    kwargs["functions"] = functions
                    
                if function_call is not None:
                    kwargs["function_call"] = function_call
                    
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                
                response = await self.client.chat.completions.create(**kwargs)
                return response
                
            except APIError as e:
                attempt += 1
                last_error = e
                logger.warning(f"API error: {str(e)}. Attempt {attempt}/{self.max_retries}")
                
                if attempt >= self.max_retries:
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
        
        logger.error(f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}")
        raise last_error