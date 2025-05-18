#!/bin/bash

# Start Flask server in the background
python -m src.api.controller &
SERVER_PID=$!

# Function to check if server is up
wait_for_server() {
    echo "Waiting for Flask server to start..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:9998/ > /dev/null 2>&1; then
            echo "Server is up!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts - Server not ready yet, waiting..."
        sleep 1
    done
    
    echo "Server failed to start after $max_attempts attempts"
    kill $SERVER_PID
    return 1
}

# Wait for server to be ready
wait_for_server || exit 1

# Make the actual request once server is ready
echo "Making API request..."
curl -X POST -i localhost:9998/api/conversations \
    -d '{ "prompt": "describe a random planet in our solar system in 10 words or less" }' \
    -H "Content-Type: application/json" || exit 1
echo

exit 0