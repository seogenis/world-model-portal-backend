#!/bin/bash

# Start both the FastAPI server and ngrok in separate terminals

# Check if terminal multiplexers are available
if command -v tmux &> /dev/null; then
    # Using tmux
    echo "Using tmux to start services..."
    tmux new-session -d -s cosmos_server 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
    tmux new-window -t cosmos_server -n ngrok './start_ngrok.sh'
    tmux attach -t cosmos_server
    exit 0
elif command -v screen &> /dev/null; then
    # Using screen
    echo "Using screen to start services..."
    screen -dmS cosmos_server bash -c 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
    screen -S cosmos_server -X screen bash -c './start_ngrok.sh'
    screen -r cosmos_server
    exit 0
else
    # No terminal multiplexers available, starting in background and foreground
    echo "Starting FastAPI server in the background..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!
    
    echo "Starting ngrok..."
    ./start_ngrok.sh
    
    # When ngrok is terminated, also terminate the server
    kill $SERVER_PID
fi