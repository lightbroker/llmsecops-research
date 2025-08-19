#!/bin/bash

# Array of model names
models=(
    "apple/OpenELM-1_1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
)

# Function to generate random 10-digit number (not needed but kept for potential future use)
# generate_random_id() {
#     echo $((RANDOM * RANDOM % 10000000000))
# }

# Create base logs directory if it doesn't exist
mkdir -p logs

# Iterate through test directories 1-5
for i in {1..5}; do
    echo "Creating test_${i} directories..."
    
    # Create range directories from 1-2 to 99-100 (50 ranges total)
    for start in {1..99..2}; do
        end=$((start + 1))
        range="${start}-${end}"
        
        # Iterate through each model
        for model in "${models[@]}"; do
            # Replace / with _ and convert to lowercase for directory name
            dir_name="${model//\//_}"
            dir_name="${dir_name}"  # Convert to lowercase
            
            # Create the full directory path
            full_path="logs/test_${i}/${dir_name}/${range}"
            mkdir -p "${full_path}"
            
            # Create placeholder JSON file
            json_file="${full_path}/_.json"
            touch "${json_file}"
            
            echo "Created: ${json_file}"
        done
    done
done

echo ""
echo "Summary:"
echo "- Created 5 test directories (test_1 through test_5)"
echo "- Created 50 range directories in each test (1-2, 3-4, ..., 99-100)"
echo "- Created 3 model directories in each range"
echo "- Created 1 JSON file in each model directory"
echo "- Total: $(find tests -name "*.json" | wc -l) JSON files created"
echo ""
echo "Example structure:"
echo "tests/logs/test_1/1-2/apple_openelm-3b-instruct/_.json"
echo "tests/logs/test_1/1-2/meta-llama_llama-3.2-3b-instruct/_.json"
echo "tests/logs/test_1/1-2/microsoft_phi-3-mini-4k-instruct/_.json"