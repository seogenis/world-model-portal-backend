import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.prompt_manager import PromptManager
from app.services.parameter_extractor import ParameterExtractor

# Sample data
TEST_PROMPT = "a couple drones flying over a burning forest"
TEST_PARAMETERS = {
    "subjects": "drones",
    "actions": "flying",
    "location": "forest",
    "atmosphere": "intense, dangerous",
    "style": "realistic"
}
TEST_UPDATE_REQUEST = "change it to a shipwrecked shore"
TEST_UPDATES = {
    "location": "shipwrecked shore",
    "atmosphere": "tense, uncertain"
}
TEST_CHANGES = [
    "Updated 'location' from 'forest' to 'shipwrecked shore'.",
    "Modified 'atmosphere' from 'intense, dangerous' to 'tense, uncertain'."
]
TEST_REGENERATED_PROMPT = "a couple drones flying over a shipwrecked shore"

@pytest.fixture
def mock_parameter_extractor():
    """Create a mock parameter extractor."""
    extractor = MagicMock(spec=ParameterExtractor)
    extractor.extract_parameters = AsyncMock(return_value=TEST_PARAMETERS)
    return extractor

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for the prompt manager."""
    with patch("openai.OpenAI") as mock_client:
        # Mock the updates response
        update_response = AsyncMock()
        update_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content=f"""
                    {{
                        "updates": {{
                            "location": "shipwrecked shore",
                            "atmosphere": "tense, uncertain"
                        }},
                        "changes_description": [
                            "Updated 'location' from 'forest' to 'shipwrecked shore'.",
                            "Modified 'atmosphere' from 'intense, dangerous' to 'tense, uncertain'."
                        ]
                    }}
                    """
                )
            )
        ]
        
        # Mock the regenerate response
        regenerate_response = AsyncMock()
        regenerate_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content=TEST_REGENERATED_PROMPT
                )
            )
        ]
        
        # Configure the mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=[
            update_response,
            regenerate_response
        ])
        
        yield mock_client

@pytest.mark.asyncio
async def test_initialize_from_prompt(mock_parameter_extractor):
    """Test prompt initialization."""
    # Create manager with mock
    manager = PromptManager(mock_parameter_extractor)
    
    # Initialize from prompt
    result = await manager.initialize_from_prompt(TEST_PROMPT)
    
    # Verify extractor was called
    mock_parameter_extractor.extract_parameters.assert_called_once_with(TEST_PROMPT)
    
    # Verify results
    assert result == TEST_PARAMETERS
    assert manager.original_prompt == TEST_PROMPT
    assert manager.current_prompt == TEST_PROMPT
    assert manager.current_parameters == TEST_PARAMETERS
    assert manager.update_history == []

@pytest.mark.asyncio
async def test_process_update_request(mock_parameter_extractor, mock_openai_client):
    """Test processing an update request."""
    # Create manager with mocks
    manager = PromptManager(mock_parameter_extractor)
    
    # Initialize manager state
    await manager.initialize_from_prompt(TEST_PROMPT)
    
    # Process update request
    params, changes = await manager.process_update_request(TEST_UPDATE_REQUEST)
    
    # Verify client was called
    mock_client_instance = mock_openai_client.return_value
    mock_client_instance.chat.completions.create.assert_called()
    
    # Verify results
    assert params == {**TEST_PARAMETERS, **TEST_UPDATES}
    assert changes == TEST_CHANGES
    assert manager.current_parameters == {**TEST_PARAMETERS, **TEST_UPDATES}
    assert len(manager.update_history) == 1
    assert manager.update_history[0]["request"] == TEST_UPDATE_REQUEST

@pytest.mark.asyncio
async def test_regenerate_prompt(mock_parameter_extractor, mock_openai_client):
    """Test regenerating a prompt."""
    # Create manager with mocks
    manager = PromptManager(mock_parameter_extractor)
    
    # Initialize manager state
    await manager.initialize_from_prompt(TEST_PROMPT)
    await manager.process_update_request(TEST_UPDATE_REQUEST)
    
    # Regenerate prompt
    result = await manager.regenerate_prompt()
    
    # Verify client was called
    mock_client_instance = mock_openai_client.return_value
    assert mock_client_instance.chat.completions.create.call_count >= 2
    
    # Verify results
    assert result == TEST_REGENERATED_PROMPT
    assert manager.current_prompt == TEST_REGENERATED_PROMPT

@pytest.mark.asyncio
async def test_minimal_prompt_changes(mock_parameter_extractor, mock_openai_client):
    """Test that prompts are minimally changed when parameters are updated."""
    # Create manager with mocks
    manager = PromptManager(mock_parameter_extractor)
    
    # Initialize with a complex prompt
    complex_prompt = "In a realistic style, a couple of advanced drones with blinking LEDs are flying high over a burning forest. The atmosphere is intense and dangerous with smoke billowing up, creating a dramatic scene."
    complex_params = {
        "subjects": "drones",
        "subject_attributes": "advanced, with blinking LEDs",
        "actions": "flying high",
        "location": "forest",
        "atmosphere": "intense, dangerous",
        "style": "realistic"
    }
    
    # Configure the mock
    mock_parameter_extractor.extract_parameters.return_value = complex_params
    
    # Initialize from prompt
    await manager.initialize_from_prompt(complex_prompt)
    
    # Mock the update response for the test
    update_response = {
        "updates": {
            "location": "shipwrecked shore",
            "atmosphere": "tense, uncertain"
        },
        "changes_description": [
            "Updated 'location' from 'forest' to 'shipwrecked shore'.",
            "Modified 'atmosphere' from 'intense, dangerous' to 'tense, uncertain'."
        ]
    }
    
    # Configure the mock response
    mock_client = mock_openai_client.return_value
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps(update_response)
    
    # Set the expected regenerated prompt that shows minimal changes
    minimally_changed_prompt = "In a realistic style, a couple of advanced drones with blinking LEDs are flying high over a shipwrecked shore. The atmosphere is tense and uncertain with debris scattered across the beach, creating a somber scene."
    mock_client.chat.completions.create.return_value.choices[0].message.content = minimally_changed_prompt
    
    # Process update request
    await manager.process_update_request("change it to a shipwrecked shore")
    result = await manager.regenerate_prompt()
    
    # Verify only the necessary parts changed
    assert "shipwrecked shore" in result
    assert "tense and uncertain" in result
    # Original elements should still be present
    assert "realistic style" in result.lower()
    assert "drones" in result.lower()
    assert "flying" in result.lower()