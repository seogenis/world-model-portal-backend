#!/bin/bash

# Cosmos Text2World Prompt Tuning API setup script
echo "Setting up Cosmos Text2World Prompt Tuning API..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating required directories..."
mkdir -p static/videos

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    echo "OPENAI_API_KEY=" > .env
    echo "NVIDIA_API_KEY=" >> .env
    echo ".env file created. Please edit it to add your API keys."
fi

# Make scripts executable
chmod +x start_ngrok.sh
chmod +x start_server_with_ngrok.sh

echo "Setup complete! You can now start the server with:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or run with ngrok:"
echo "  ./start_server_with_ngrok.sh"