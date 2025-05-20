#!/bin/bash
set -e  # Exit on error

cd $GITHUB_WORKSPACE

echo "Waiting for API to be ready..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
  echo "Health check attempt $attempt of $max_attempts..."
  if curl -s -f -i http://localhost:9999/ > logs/health_check_$attempt.log 2>&1; then
    echo "Health check succeeded"
    break
  else
    echo "Health check failed, waiting 5 seconds..."
    sleep 5
    attempt=$((attempt+1))
  fi
done

if [ $attempt -gt $max_attempts ]; then
  echo "API failed to start after $max_attempts attempts"
  cat logs/api.log
  exit 1
fi