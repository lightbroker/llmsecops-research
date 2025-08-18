#!/bin/bash

# Array of model names
models=(
    "apple/OpenELM-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
)

# Create base integration directory if it doesn't exist

# Iterate through test directories 1-5
for i in {0..4}; do
    # Iterate through each model
    for model in "${models[@]}"; do
        # Replace / with 
        dir_name="${model//\//_}"
        
        # Create the directory structure
        mkdir -p "logs/test_${i}/${dir_name}"
        
        echo "Created: logs/test_${i}/${dir_name}"
    done
done

echo "All directories created successfully!"