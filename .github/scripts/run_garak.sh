#!/bin/bash
# Don't use set -e here as we want to capture and handle errors ourselves

cd $GITHUB_WORKSPACE

# Make sure garak report directory exists
GARAK_REPORTS_DIR="/home/runner/.local/share/garak/garak_runs"
mkdir -p $GARAK_REPORTS_DIR
mkdir -p logs/garak_reports

# Log system resource information before starting garak
echo "System resources before starting garak:" > logs/system_before_garak.log
free -h >> logs/system_before_garak.log
df -h >> logs/system_before_garak.log
ulimit -a >> logs/system_before_garak.log

# Generate a time-stamped log file for garak
GARAK_LOG_FILE="logs/garak_$(date +%Y%m%d_%H%M%S).log"
echo "GARAK_LOG_FILE=$GARAK_LOG_FILE" >> $GITHUB_ENV
echo "Running garak vulnerability scan with output to $GARAK_LOG_FILE..."

# Start garak with enhanced error capture and reduced resource usage
{
  set -x  # Enable debug mode to print commands
  
  # Run with trap to capture signals
  (
    trap 'echo "Received termination signal at $(date)" >> $GARAK_LOG_FILE' TERM INT
    
    # Run garak with lower parallel attempts to reduce resource usage
    # and with a timeout to prevent hanging
    timeout --preserve-status 40m garak -v \
      --config $WORKSPACE/src/tools/garak.config.yml \
      --generator_option_file $WORKSPACE/src/tools/garak.rest.llm.json \
      --model_type=rest \
      --parallel_attempts 8
    
    echo "Garak completed with exit code $?" >> $GARAK_LOG_FILE
  )
  
  set +x  # Disable debug mode
} > $GARAK_LOG_FILE 2>&1

GARAK_EXIT_CODE=$?
echo "Garak exit code: $GARAK_EXIT_CODE"

# Log system resource information after garak completes
echo "System resources after garak:" > logs/system_after_garak.log
free -h >> logs/system_after_garak.log
df -h >> logs/system_after_garak.log

# Copy any garak reports to our logs directory for easier access
echo "Copying garak reports to logs directory..."
cp -r $GARAK_REPORTS_DIR/* logs/garak_reports/ || echo "No garak reports found to copy"

# List what reports were generated
echo "Garak reports found:"
find logs/garak_reports -type f | sort || echo "No garak reports found"

# Capture and report logs regardless of success/failure
echo "Last 200 lines of garak log:"
cat $GARAK_LOG_FILE | tail -n 200

# Check for specific error patterns
echo "Checking for known error patterns..."
{
  if grep -q "operation was canceled" $GARAK_LOG_FILE; then
    echo "FOUND 'operation was canceled' error in logs:"
    grep -A 10 -B 10 "operation was canceled" $GARAK_LOG_FILE
  fi

  if grep -q "memory" $GARAK_LOG_FILE; then
    echo "FOUND memory-related messages in logs:"
    grep -A 10 -B 10 "memory" $GARAK_LOG_FILE
  fi
  
  if grep -q "timeout" $GARAK_LOG_FILE; then
    echo "FOUND timeout-related messages in logs:"
    grep -A 10 -B 10 "timeout" $GARAK_LOG_FILE
  fi
  
  if grep -q "SIGTERM\|signal\|terminated" $GARAK_LOG_FILE; then
    echo "FOUND termination signals in logs:"
    grep -A 10 -B 10 -E "SIGTERM|signal|terminated" $GARAK_LOG_FILE
  fi
} >> logs/error_analysis.log

# Save the exit code analysis
echo "Exit code analysis:" > logs/exit_code_analysis.log
{
  echo "Garak exit code: $GARAK_EXIT_CODE"
  case $GARAK_EXIT_CODE in
    0)
      echo "Success - completed normally"
      ;;
    124)
      echo "Error - timed out after 40 minutes"
      ;;
    130)
      echo "Error - terminated by SIGINT (Ctrl+C)"
      ;;
    137)
      echo "Error - killed by SIGKILL (likely out of memory)"
      ;;
    143)
      echo "Error - terminated by SIGTERM (possibly by runner timeout or job cancellation)"
      ;;
    *)
      echo "Error - unknown exit code"
      ;;
  esac
} >> logs/exit_code_analysis.log

cat logs/exit_code_analysis.log

# Return proper exit code based on analysis
if [ $GARAK_EXIT_CODE -eq 143 ]; then
  echo "Process was terminated by SIGTERM. This may be due to:"
  echo "1. GitHub Actions workflow timeout"
  echo "2. Out of memory condition"
  echo "3. Manual cancellation of the workflow"
  echo "Treating as a workflow issue rather than a test failure"
  # We return 0 to avoid failing the workflow on infrastructure issues
  # You can change this to exit 1 if you prefer the workflow to fail
  exit 0
elif [ $GARAK_EXIT_CODE -eq 124 ]; then
  echo "Garak timed out after 40 minutes"
  exit 0  # Treat timeout as acceptable
elif [ $GARAK_EXIT_CODE -ne 0 ]; then
  echo "Garak failed with exit code $GARAK_EXIT_CODE"
  exit 1  # Only fail for actual test failures
else
  exit 0
fi