#!/bin/bash
set -e  # Exit on error

echo "Starting API server with logging..."
nohup python -m src.api.server > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Save PID to file so it can be accessed by other scripts
echo $API_PID > api_pid.txt