#!/bin/bash
set -e  # Exit on error

cd $GITHUB_WORKSPACE

echo "Starting API server with logging..."
nohup uvicorn src.api.http_api:app --host 0.0.0.0 --port 9999 > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Save PID to file so it can be accessed by other scripts
echo $API_PID > api_pid.txt