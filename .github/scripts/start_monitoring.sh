#!/bin/bash

echo "Starting system monitoring..."

cd $GITHUB_WORKSPACE

# Read API PID from file
API_PID=$(cat api_pid.txt)
echo "Monitoring API process with PID: $API_PID"

# Save monitoring PID to file for later cleanup
echo $$ > $MONITOR_PID_FILE

while true; do
  date >> logs/system_monitor.log
  echo "Memory usage:" >> logs/system_monitor.log
  free -m >> logs/system_monitor.log
  echo "Process info:" >> logs/system_monitor.log
  ps aux | grep -E 'python|garak' >> logs/system_monitor.log
  echo "Network connections:" >> logs/system_monitor.log
  netstat -tulpn | grep python >> logs/system_monitor.log 2>/dev/null || echo "No network connections found" >> logs/system_monitor.log
  echo "API process status:" >> logs/system_monitor.log
  if ps -p $API_PID > /dev/null; then
    echo "API process is running" >> logs/system_monitor.log
  else
    echo "API process is NOT running!" >> logs/system_monitor.log
  fi
  echo "-------------------" >> logs/system_monitor.log
  sleep 10
done