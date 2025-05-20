#!/bin/bash

echo "Cleaning up processes..."

# Kill the monitoring process if it exists
if [ -f "$MONITOR_PID_FILE" ]; then
  MONITOR_PID=$(cat $MONITOR_PID_FILE)
  echo "Stopping monitoring process with PID: $MONITOR_PID"
  kill $MONITOR_PID 2>/dev/null || echo "Monitor process already stopped"
  rm $MONITOR_PID_FILE
fi

# Kill the API process if it exists
if [ -f "$API_PID_FILE" ]; then
  API_PID=$(cat $API_PID_FILE)
  echo "Stopping API process with PID: $API_PID"
  kill $API_PID 2>/dev/null || echo "API process already stopped"
  rm $API_PID_FILE
fi

echo "Cleanup complete"