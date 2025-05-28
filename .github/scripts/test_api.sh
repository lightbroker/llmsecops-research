#!/bin/bash
# Local-only usage: ./test_api.sh --local

set -e  # Exit on error

# Parse command line arguments
LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$LOCAL" = false ]; then
    cd $GITHUB_WORKSPACE
fi

echo "Making API request..."

# Wait for server to start and verify it's running
max_retries=200
retry_count=0
server_ready=false

while [ $retry_count -lt $max_retries ] && [ "$server_ready" = false ]; do
  echo "Waiting for server to start (attempt $retry_count/$max_retries)..."
  if curl -s -o /dev/null -w "%{http_code}" localhost:9999 > /dev/null 2>&1; then
    server_ready=true
    echo "Server is running"
  else
    sleep 2
    retry_count=$((retry_count + 1))
  fi
done

if [ "$server_ready" = false ]; then
  echo "::error::Server failed to start after $max_retries attempts"
  exit 1
fi