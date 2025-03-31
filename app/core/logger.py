import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("cosmos-prompt-tuner")


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logger


def log_parameters(parameters: Dict[str, Any], prefix: str = "") -> None:
    """Log parameter values in a structured way."""
    logger.info(f"{prefix} Parameters:")
    for key, value in parameters.items():
        logger.info(f"  {key}: {value}")


def log_parameter_changes(old_params: Dict[str, Any], new_params: Dict[str, Any]) -> None:
    """Log changes between two sets of parameters."""
    changed_keys = []
    
    # Check for changes to existing keys
    for key in old_params:
        if key in new_params and old_params[key] != new_params[key]:
            logger.info(f"Parameter changed: '{key}' from '{old_params[key]}' to '{new_params[key]}'")
            changed_keys.append(key)
    
    # Check for new keys
    for key in new_params:
        if key not in old_params:
            logger.info(f"Parameter added: '{key}' with value '{new_params[key]}'")
            changed_keys.append(key)
    
    # Check for removed keys
    for key in old_params:
        if key not in new_params:
            logger.info(f"Parameter removed: '{key}' (was '{old_params[key]}')")
            changed_keys.append(key)
    
    if not changed_keys:
        logger.info("No parameter changes detected")


def log_prompt_change(original: str, updated: str) -> None:
    """Log how a prompt was changed."""
    logger.info(f"Original prompt: {original[:50]}...")
    logger.info(f"Updated prompt: {updated[:50]}...")


def log_operation(operation: str, details: str = "", level: str = "info") -> None:
    """Log an operation with optional details."""
    log_message = f"Operation: {operation}"
    if details:
        log_message += f" | {details}"
        
    if level.lower() == "debug":
        logger.debug(log_message)
    elif level.lower() == "warning":
        logger.warning(log_message)
    elif level.lower() == "error":
        logger.error(log_message)
    else:
        logger.info(log_message)