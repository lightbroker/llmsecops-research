#!/usr/bin/bash

# create Python virtual environment
virtualenv --python="/usr/bin/python3.12" .env
source .env/bin/activate

# the ONNX model/data require git Large File System support
git lfs install

# get the system-under-test LLM dependencies from HuggingFace / Microsoft
pip3.12 install huggingface-hub[cli]
cd ./tests/llm
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
pip3.12 install onnxruntime-genai

if ! [[ -e ./phi3-qa.py ]]
then
    curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py -o ./phi3-qa.py
fi

python3.12 ./phi3-qa.py -m ./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -e cpu -v