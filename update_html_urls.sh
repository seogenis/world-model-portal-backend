#!/bin/bash

# This script updates the API_BASE_URL in HTML files with a new URL

if [ -z "$1" ]; then
    echo "Error: Please provide the Ngrok URL as an argument."
    echo "Usage: ./update_html_urls.sh https://your-ngrok-url.ngrok.io"
    exit 1
fi

NEW_URL=$1

# Remove trailing slash if present
NEW_URL=${NEW_URL%/}

echo "Updating HTML files with new API URL: $NEW_URL"

# List of HTML files to update
HTML_FILES=(
    "/Users/seanwu/Documents/Programming/WorldModelPortalServer/agent_prompt_tuner/video_player.html"
    "/Users/seanwu/Documents/Programming/WorldModelPortalServer/agent_prompt_tuner/prompt_enhancer.html"
    "/Users/seanwu/Documents/Programming/WorldModelPortalServer/agent_prompt_tuner/prompt_variations.html"
    "/Users/seanwu/Documents/Programming/WorldModelPortalServer/agent_prompt_tuner/full_interface.html"
)

for file in "${HTML_FILES[@]}"; do
    if [ -f "$file" ]; then
        # Replace any URL in the API_BASE_URL line with the new URL
        sed -i.bak 's|const API_BASE_URL = '\''http://[^'\'']*'\''|const API_BASE_URL = '\'''"$NEW_URL"''\''|g' "$file"
        echo "Updated: $file"
    else
        echo "Warning: File not found: $file"
    fi
done

echo ""
echo "All HTML files have been updated with the new API URL."
echo "You can now open any of these HTML files on another computer to access your API."