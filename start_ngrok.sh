#!/bin/bash

echo "Starting ngrok tunnel to your FastAPI server on port 8000..."
echo "The public URL will be displayed below."
echo "Use this URL to access your API from any network."
echo ""
echo "IMPORTANT: Replace the API_BASE_URL in your HTML files with the ngrok URL!"
echo "Look for the 'Forwarding' line in the output below."
echo ""
echo "Press Ctrl+C to stop the tunnel when you're done."

# Start ngrok
ngrok http 8000