#!/usr/bin/bash
# Local-only usage: ./run.sh --local

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

if [ "$LOCAL" = true ]; then
    # create Python virtual environment
    python3.12 -m venv .env
    source .env/bin/activate
fi

# the ONNX model/data require git Large File System support
git lfs install

# install Python dependencies
pip install -r ./requirements.txt

# environment variables
export GARAK_PROBE='dan.DAN_Jailbreak'
export PROMPT_TEMPLATES_DIR="./infrastructure/prompt_templates"
export MODEL_BASE_DIR="./infrastructure/foundation_model"
export MODEL_CPU_DIR="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
MODEL_DATA_FILENAME="phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data"
MODEL_DATA_FILEPATH="$MODEL_BASE_DIR/$MODEL_CPU_DIR/$MODEL_DATA_FILENAME"


# get foundation model dependencies from HuggingFace / Microsoft
if [ ! -f "$MODEL_DATA_FILEPATH" ]; then
    echo "Downloading foundation model..."
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
        --include "$MODEL_CPU_DIR/*" \
        --local-dir $MODEL_BASE_DIR
else
    echo "Foundation model files exist at: $MODEL_DATA_FILEPATH"
fi

python -m src.text_generation.entrypoints.__main__