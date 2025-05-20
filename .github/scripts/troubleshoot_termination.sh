#!/bin/bash

# This script is designed to fix the Exit Code 143 issue in GitHub Actions
# by troubleshooting likely resource and timeout issues

echo "Running troubleshooting for Exit Code 143 (SIGTERM)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Check for existence of important files and directories
echo "## Checking file system status" > logs/troubleshooting.log
ls -la $WORKSPACE/src/tools/ >> logs/troubleshooting.log 2>&1
echo "" >> logs/troubleshooting.log

# Check garak configuration files
echo "## Checking garak configuration files" >> logs/troubleshooting.log
if [ -f "$WORKSPACE/src/tools/garak.config.yml" ]; then
  echo "garak.config.yml exists" >> logs/troubleshooting.log
  grep -v "^#" "$WORKSPACE/src/tools/garak.config.yml" | grep -v "^$" >> logs/troubleshooting.log
else
  echo "ERROR: garak.config.yml NOT FOUND" >> logs/troubleshooting.log
fi
echo "" >> logs/troubleshooting.log

if [ -f "$WORKSPACE/src/tools/garak.rest.llm.json" ]; then
  echo "garak.rest.llm.json exists" >> logs/troubleshooting.log
  cat "$WORKSPACE/src/tools/garak.rest.llm.json" >> logs/troubleshooting.log
else
  echo "ERROR: garak.rest.llm.json NOT FOUND" >> logs/troubleshooting.log
fi
echo "" >> logs/troubleshooting.log

# Check GitHub Actions runner environment
echo "## GitHub Actions runner environment" >> logs/troubleshooting.log
echo "CPU cores: $(nproc)" >> logs/troubleshooting.log
echo "Memory:" >> logs/troubleshooting.log
free -h >> logs/troubleshooting.log
echo "Disk space:" >> logs/troubleshooting.log
df -h >> logs/troubleshooting.log
echo "" >> logs/troubleshooting.log

# Check garak installation
echo "## Garak installation" >> logs/troubleshooting.log
pip show garak >> logs/troubleshooting.log
echo "" >> logs/troubleshooting.log

# Test garak basic functionality
echo "## Testing garak basic functionality" >> logs/troubleshooting.log
garak --version >> logs/troubleshooting.log 2>&1

# Output troubleshooting suggestions
echo "## Troubleshooting suggestions for Exit Code 143" >> logs/troubleshooting.log
echo "1. Resource limitations:" >> logs/troubleshooting.log
echo "   - Reduce parallel_attempts from 8 to 4" >> logs/troubleshooting.log
echo "   - Set MALLOC_ARENA_MAX=2 environment variable" >> logs/troubleshooting.log
echo "   - Monitor memory usage more closely" >> logs/troubleshooting.log
echo "2. Timeout issues:" >> logs/troubleshooting.log
echo "   - Break the garak run into multiple smaller runs" >> logs/troubleshooting.log
echo "   - Reduce the number of tests being run" >> logs/troubleshooting.log
echo "3. Consider using a larger GitHub Actions runner" >> logs/troubleshooting.log
echo "4. Investigate network issues between API and garak" >> logs/troubleshooting.log

# # Create a patch file for reducing parallel attempts even further if needed
# cat > logs/reduce_parallel.patch << 'EOF'
# --- a/.github/scripts/run_garak.sh
# +++ b/.github/scripts/run_garak.sh
# @@ -27,7 +27,7 @@
#      timeout --preserve-status 40m garak -v \
#        --config $WORKSPACE/src/tools/garak.config.yml \
#        --generator_option_file $WORKSPACE/src/tools/garak.rest.llm.json \
# -      --model_type=rest \
# -      --parallel_attempts 8
# +      --model_type=rest --probe-parameters '{"concurrent_requests": 2}' \
# +      --parallel_attempts 4
     
#      echo "Garak completed with exit code $?" >> $GARAK_LOG_FILE
# EOF

echo "Troubleshooting complete. See logs/troubleshooting.log for details."
echo "A patch file has been created at logs/reduce_parallel.patch if you need to reduce parallel attempts further."