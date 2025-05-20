#!/bin/bash
set -e  # Exit on error

cd $GITHUB_WORKSPACE

echo "Making API request..."
curl -X POST -i http://localhost:9999/api/conversations \
  -d '{ "prompt": "describe a random planet in our solar system in 10 words or less" }' \
  -H "Content-Type: application/json" > logs/test_request.log 2>&1

if [ $? -ne 0 ]; then
  echo "Test API request failed"
  cat logs/test_request.log
  exit 1
else
  echo "Test API request succeeded"
  cat logs/test_request.log
fi