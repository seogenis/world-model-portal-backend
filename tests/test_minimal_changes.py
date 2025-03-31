import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager
from app.core.logger import get_logger

logger = get_logger()

# Sample OpenAI responses for mocking
MOCK_EXTRACT_RESPONSE = {
    "subjects": "drones",
    "actions": "flying",
    "location": "forest",
    "atmosphere": "intense, dangerous",
    "style": "realistic"
}

MOCK_UPDATE_RESPONSE = {
    "updates": {
        "location": "shipwrecked shore",
        "atmosphere": "tense, uncertain"
    },
    "changes_description": [
        "Updated 'location' from 'forest' to 'shipwrecked shore'.",
        "Modified 'atmosphere' from 'intense, dangerous' to 'tense, uncertain'."
    ]
}

MOCK_REGENERATED_PROMPT = "a couple drones flying over a shipwrecked shore"


class MockOpenAI:
    def __init__(self, api_key=None):
        self.chat = MockChatCompletions()


class MockChatCompletions:
    def create(self, model=None, messages=None, tools=None):
        # Check which API call is being made based on the messages
        message_content = messages[1]["content"] if messages and len(messages) > 1 else ""
        
        if "Extract parameters from" in message_content:
            return MockResponse([MockChoice(MockMessage(content=json.dumps(MOCK_EXTRACT_RESPONSE)))])
        elif "Current parameters" in message_content and "User request: change it to a shipwrecked shore" in message_content:
            return MockResponse([MockChoice(MockMessage(content=json.dumps(MOCK_UPDATE_RESPONSE)))])
        elif "Generate a new coherent prompt" in message_content:
            return MockResponse([MockChoice(MockMessage(content=MOCK_REGENERATED_PROMPT))])
        
        # Default fallback
        return MockResponse([MockChoice(MockMessage(content="{}"))])


class MockResponse:
    def __init__(self, choices):
        self.choices = choices


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockMessage:
    def __init__(self, content=None, function_call=None):
        self.content = content
        self.function_call = function_call


async def test_minimal_changes():
    """Test the minimal changes feature with the drone example."""
    # Override the OpenAI client
    import openai
    openai.OpenAI = MockOpenAI
    
    # Create the components
    extractor = ParameterExtractor("fake-api-key")
    manager = PromptManager(extractor)
    
    print("\n=== TESTING MINIMAL PROMPT CHANGES ===")
    
    # Step 1: Initialize with the first prompt
    initial_prompt = "a couple drones flying over a burning forest"
    print(f"\nInitial prompt: '{initial_prompt}'")
    
    parameters = await manager.initialize_from_prompt(initial_prompt)
    print("\nExtracted parameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")
    
    # Step 2: Process the update request
    update_request = "change it to a shipwrecked shore"
    print(f"\nUser request: '{update_request}'")
    
    updated_params, changes = await manager.process_update_request(update_request)
    print("\nChanges made:")
    for change in changes:
        print(f"  • {change}")
    
    print("\nUpdated parameters:")
    for key, value in updated_params.items():
        print(f"  {key}: {value}")
    
    # Step 3: Generate the new prompt
    new_prompt = await manager.regenerate_prompt()
    print(f"\nNew prompt: '{new_prompt}'")
    
    # Verify the prompt was minimally changed
    assert "drones" in new_prompt  # Subject preserved
    assert "flying" in new_prompt  # Action preserved
    assert "shipwrecked shore" in new_prompt  # Location updated
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_minimal_changes())