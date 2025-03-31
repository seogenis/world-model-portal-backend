import asyncio
import json
import httpx
import sys
from typing import Dict, Any, List, Optional

BASE_URL = "http://localhost:8000/api"

# Command-line arguments handling
COMMANDS = {
    "interactive": "Start interactive mode",
    "enhance": "Enhance a rough prompt with descriptive details",
    "variations": "Generate variations of selected prompts",
    "initialize": "Initialize with a prompt and extract parameters",
    "update": "Update an existing prompt based on a request", 
    "history": "View prompt history"
}


async def make_request(endpoint: str, method: str = "POST", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the API."""
    async with httpx.AsyncClient() as client:
        if method.upper() == "POST":
            response = await client.post(f"{BASE_URL}{endpoint}", json=data)
        else:
            response = await client.get(f"{BASE_URL}{endpoint}")
            
        response.raise_for_status()
        return response.json()


def print_parameters(parameters: Dict[str, Any]) -> None:
    """Print parameters in a formatted way."""
    print("\n=== Parameters ===")
    for param, value in parameters.items():
        print(f"{param}: {value}")


def print_changes(changes: List[str]) -> None:
    """Print changes in a formatted way."""
    if not changes:
        return
        
    print("\n=== Changes Made ===")
    for change in changes:
        print(f"• {change}")


def print_prompt(prompt: str) -> None:
    """Print the prompt in a formatted way."""
    print("\n=== Prompt ===")
    print(prompt)


async def enhance_prompt(rough_prompt: str) -> None:
    """Enhance a rough prompt with descriptive details."""
    try:
        print("\nEnhancing prompt...")
        result = await make_request("/enhance", "POST", {"rough_prompt": rough_prompt})
        
        print("\n=== Original Prompt ===")
        print(result["original_prompt"])
        
        print("\n=== Enhanced Prompt ===")
        print(result["enhanced_prompt"])
        
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")


async def generate_variations(selected_indices: List[int], total_count: int = 8) -> None:
    """Generate variations of selected prompts."""
    try:
        print("\nGenerating prompt variations...")
        result = await make_request("/generate-variations", "POST", {
            "selected_indices": selected_indices,
            "total_count": total_count
        })
        
        print("\n=== Generated Variations ===")
        for i, prompt in enumerate(result["prompts"]):
            is_selected = i in [result["selected_indices"].index(idx) if i < len(result["selected_indices"]) else -1 
                               for idx in result["selected_indices"]]
            
            marker = "✓" if is_selected else " "
            print(f"\n[{marker}] Prompt #{i+1}:")
            print(prompt)
        
    except Exception as e:
        print(f"Error generating variations: {str(e)}")


async def get_history() -> None:
    """Get all prompt history."""
    try:
        result = await make_request("/history", "GET")
        
        print("\n=== Prompt History ===")
        print(f"Number of prompts: {len(result['history'])}")
        
        for i, item in enumerate(result['history']):
            print(f"\nPrompt #{i}:")
            print(f"Description: {item['description']}")
            print(f"Prompt: {item['prompt'][:100]}...")
            
    except Exception as e:
        print(f"Error getting prompt history: {str(e)}")


async def interactive_mode():
    """Run the interactive mode."""
    print("=== Cosmos Prompt Tuner CLI Simulator ===")
    print("Enter an initial prompt to begin:")
    initial_prompt = input("> ")
    
    try:
        # Initialize with the prompt
        print("\nProcessing initial prompt...")
        result = await make_request("/initialize", "POST", {"prompt": initial_prompt})
        
        print_parameters(result["parameters"])
        print_prompt(result["prompt"])
        
        # Interactive loop
        while True:
            print("\nEnter command or modification request:")
            print("Commands: 'quit', 'history', 'enhance', 'variations'")
            user_input = input("> ")
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
                
            elif user_input.lower() == "history":
                await get_history()
                
            elif user_input.lower() == "enhance":
                print("Enter a rough prompt to enhance:")
                rough_prompt = input("> ")
                await enhance_prompt(rough_prompt)
                
            elif user_input.lower() == "variations":
                print("Enter indices of prompts to use as base (comma-separated):")
                indices_input = input("Indices> ")
                try:
                    indices = [int(i.strip()) for i in indices_input.split(",")]
                    print("Enter total number of variations to generate (default: 8):")
                    count_input = input("Count [8]> ")
                    count = int(count_input) if count_input.strip() else 8
                    await generate_variations(indices, count)
                except ValueError as e:
                    print(f"Error parsing input: {str(e)}")
            
            else:
                # Treat as update request
                print("Processing your request...")
                result = await make_request("/update", "POST", {"user_request": user_input})
                
                print_changes(result["changes"])
                print_parameters(result["parameters"])
                print_prompt(result["prompt"])
            
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function that handles command-line arguments."""
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        print("=== Cosmos Prompt Tuner CLI ===")
        print("Available commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:<12} - {desc}")
        return
        
    command = sys.argv[1].lower()
    
    if command == "interactive":
        await interactive_mode()
        
    elif command == "enhance" and len(sys.argv) >= 3:
        rough_prompt = sys.argv[2]
        await enhance_prompt(rough_prompt)
        
    elif command == "variations" and len(sys.argv) >= 3:
        try:
            indices = [int(i.strip()) for i in sys.argv[2].split(",")]
            total_count = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
            await generate_variations(indices, total_count)
        except ValueError:
            print("Error: Invalid indices format. Use comma-separated integers.")
            
    elif command == "initialize" and len(sys.argv) >= 3:
        prompt = sys.argv[2]
        try:
            result = await make_request("/initialize", "POST", {"prompt": prompt})
            print_parameters(result["parameters"])
            print_prompt(result["prompt"])
        except Exception as e:
            print(f"Error initializing prompt: {str(e)}")
            
    elif command == "update" and len(sys.argv) >= 3:
        user_request = sys.argv[2]
        try:
            result = await make_request("/update", "POST", {"user_request": user_request})
            print_changes(result["changes"])
            print_parameters(result["parameters"])
            print_prompt(result["prompt"])
        except Exception as e:
            print(f"Error updating prompt: {str(e)}")
            
    elif command == "history":
        await get_history()
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'help' to see available commands.")


def run_async():
    """Run the async main function with proper handling for different Python versions."""
    try:
        asyncio.run(main())
    except AttributeError:
        # For older Python versions that don't have asyncio.run
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.close()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    run_async()