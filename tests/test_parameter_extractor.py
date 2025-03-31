import pytest
import os
from unittest.mock import AsyncMock, patch
from app.services.parameter_extractor import ParameterExtractor

# Sample test prompt
TEST_PROMPT = """The teal robot is cooking food in a kitchen. Steam rises from a simmering pot as the robot 
chops vegetables on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints 
of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered 
measuring spoons and a half-empty bottle of olive oil."""

# Sample expected response from OpenAI API
MOCK_RESPONSE = AsyncMock()
MOCK_RESPONSE.choices = [
    AsyncMock(
        message=AsyncMock(
            function_call=AsyncMock(
                arguments="""
                {
                    "parameters": {
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
                }
                """
            )
        )
    )
]

@pytest.mark.asyncio
@patch("openai.OpenAI")
async def test_extract_parameters(mock_openai_client):
    """Test parameter extraction from a prompt."""
    # Configure mock
    mock_client_instance = mock_openai_client.return_value
    mock_client_instance.chat.completions.create = AsyncMock(return_value=MOCK_RESPONSE)
    
    # Create extractor with mock client
    extractor = ParameterExtractor(api_key="test_key")
    
    # Extract parameters
    result = await extractor.extract_parameters(TEST_PROMPT)
    
    # Verify call was made correctly
    mock_client_instance.chat.completions.create.assert_called_once()
    
    # Verify results
    assert isinstance(result, dict)
    assert "subject" in result
    assert result["subject"] == "teal robot"
    assert result["location"] == "kitchen"
    assert "prop_1" in result
    assert "atmosphere" in result