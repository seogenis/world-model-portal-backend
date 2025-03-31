#!/bin/bash

# Script to update the ngrok URLs in all frontend examples

if [ $# -ne 1 ]; then
    echo "Usage: $0 <ngrok-url>"
    echo "Example: $0 https://abcd-123-45-67-89.ngrok.io"
    exit 1
fi

NGROK_URL=$1

# Remove trailing slash if present
NGROK_URL=${NGROK_URL%/}

echo "Updating frontend examples with ngrok URL: $NGROK_URL"

# Get all HTML files in the frontend_examples directory
HTML_FILES=$(find frontend_examples -name "*.html")

for file in $HTML_FILES; do
    echo "Updating $file..."
    
    # Replace the placeholder URL with the actual ngrok URL
    # Different files might have different placeholders, so we handle multiple cases
    
    # Replace 'YOUR_NGROK_URL.ngrok.io' with the actual domain (without https://)
    DOMAIN=$(echo $NGROK_URL | sed 's|https://||')
    sed -i'.bak' "s|YOUR_NGROK_URL.ngrok.io|$DOMAIN|g" "$file"
    
    # Replace 'https://YOUR_NGROK_URL.ngrok.io' with the full URL
    sed -i'.bak' "s|https://YOUR_NGROK_URL.ngrok.io|$NGROK_URL|g" "$file"
    
    # Replace other possible patterns as needed
    sed -i'.bak' "s|const API_BASE_URL = .*|const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000/api' : '$NGROK_URL/api';|g" "$file"
    
done

# Clean up backup files
find frontend_examples -name "*.bak" -delete

echo "Frontend examples updated successfully!"
echo "You can now access the examples at: $NGROK_URL/frontend_examples/*.html"