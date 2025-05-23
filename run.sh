#!/usr/bin/bash

# create Python virtual environment
python3.12 -m venv .env
source .env/bin/activate

# the ONNX model/data require git Large File System support
git lfs install

# pip install huggingface-hub[cli]

# # get foundation model dependencies from HuggingFace / Microsoft
# huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
#     --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
#     --local-dir ./infrastructure/foundation_model

python -m src.text_generation.entrypoints.server

