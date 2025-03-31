import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager

# Sample data
TEST_PROMPT = "The teal robot is cooking food in a kitchen."
TEST_PARAMETERS = {
    "subject": "teal robot",
    "subject_action": "cooking food",
    "location": "kitchen"
}
TEST_UPDATE_REQUEST = "Change the robot to blue"
TEST_UPDATES = {
    "subject": "blue robot"
}
TEST_CHANGES = [
    "Changed robot color from teal to blue"
]
TEST_REGENERATED_PROMPT = "The blue robot is cooking food in a kitchen."

# New test constants
TEST_ROUGH_PROMPT = "Three drones flying over earthquake mountains cabin"
TEST_ENHANCED_PROMPT = "At dawn, three matte-black quad-rotor drones hover over a landslide-stricken mountain slope. Their red and green lights pulse as pale orange sunlight reveals devastation—fallen trees, debris, a collapsed cabin, and twisted metal."
TEST_PROMPT_HISTORY = [
    {
        "prompt": TEST_PROMPT,
        "parameters": TEST_PARAMETERS,
        "description": "Initial prompt"
    },
    {
        "prompt": TEST_REGENERATED_PROMPT,
        "parameters": {**TEST_PARAMETERS, **TEST_UPDATES},
        "description": TEST_UPDATE_REQUEST
    }
]
TEST_VARIATIONS = [
    "The blue robot is cooking food in a kitchen.", 
    "The blue robot is cooking food in a modern living room.",
    "The blue robot is serving drinks at a beach bar."
]

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager for testing."""
    with patch("app.api.routes.get_prompt_manager") as mock_get_manager:
        # Create mock manager
        manager = MagicMock(spec=PromptManager)
        
        # Configure mock methods
        manager.initialize_from_prompt = AsyncMock(return_value=TEST_PARAMETERS)
        manager.process_update_request = AsyncMock(return_value=(
            {**TEST_PARAMETERS, **TEST_UPDATES},
            TEST_CHANGES
        ))
        manager.regenerate_prompt = AsyncMock(return_value=TEST_REGENERATED_PROMPT)
        manager.enhance_prompt = AsyncMock(return_value=TEST_ENHANCED_PROMPT)
        manager.generate_prompt_variations = AsyncMock(return_value=TEST_VARIATIONS)
        manager.current_parameters = {**TEST_PARAMETERS, **TEST_UPDATES}
        manager.original_prompt = TEST_PROMPT
        manager.current_prompt = TEST_REGENERATED_PROMPT
        manager.get_history = MagicMock(return_value=[
            {
                "request": TEST_UPDATE_REQUEST,
                "updates": TEST_UPDATES,
                "changes": TEST_CHANGES
            }
        ])
        manager.get_prompt_history = MagicMock(return_value=TEST_PROMPT_HISTORY)
        
        # Return the mock
        mock_get_manager.return_value = manager
        yield manager

def test_initialize_prompt(mock_prompt_manager):
    """Test the initialize endpoint."""
    response = client.post(
        "/api/initialize",
        json={"prompt": TEST_PROMPT}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["parameters"] == TEST_PARAMETERS
    assert data["prompt"] == TEST_PROMPT
    assert data["changes"] == []
    
    # Verify manager was called
    mock_prompt_manager.initialize_from_prompt.assert_called_once_with(TEST_PROMPT)

def test_update_prompt(mock_prompt_manager):
    """Test the update endpoint."""
    response = client.post(
        "/api/update",
        json={"user_request": TEST_UPDATE_REQUEST}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["parameters"] == {**TEST_PARAMETERS, **TEST_UPDATES}
    assert data["prompt"] == TEST_REGENERATED_PROMPT
    assert data["changes"] == TEST_CHANGES
    
    # Verify manager was called
    mock_prompt_manager.process_update_request.assert_called_once_with(TEST_UPDATE_REQUEST)
    mock_prompt_manager.regenerate_prompt.assert_called_once()

def test_get_parameters(mock_prompt_manager):
    """Test the get parameters endpoint."""
    response = client.get("/api/parameters")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data == {**TEST_PARAMETERS, **TEST_UPDATES}

def test_enhance_prompt(mock_prompt_manager):
    """Test the enhance prompt endpoint."""
    response = client.post(
        "/api/enhance",
        json={"rough_prompt": TEST_ROUGH_PROMPT}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["original_prompt"] == TEST_ROUGH_PROMPT
    assert data["enhanced_prompt"] == TEST_ENHANCED_PROMPT
    
    # Verify manager was called
    mock_prompt_manager.enhance_prompt.assert_called_once_with(TEST_ROUGH_PROMPT)


def test_get_history(mock_prompt_manager):
    """Test the get history endpoint."""
    response = client.get("/api/history")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert len(data["history"]) == 2
    assert data["history"][0]["prompt"] == TEST_PROMPT
    assert data["history"][1]["prompt"] == TEST_REGENERATED_PROMPT
    
    # Verify manager was called
    mock_prompt_manager.get_prompt_history.assert_called_once()


def test_generate_variations(mock_prompt_manager):
    """Test the generate variations endpoint."""
    response = client.post(
        "/api/generate-variations",
        json={"selected_indices": [0], "total_count": 3}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "prompts" in data
    assert "selected_indices" in data
    assert data["prompts"] == TEST_VARIATIONS
    assert data["selected_indices"] == [0]
    
    # Verify manager was called
    mock_prompt_manager.generate_prompt_variations.assert_called_once_with([0], 3)