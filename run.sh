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
# git lfs install

# install Python dependencies
# pip install -r ./requirements.txt

# environment variables


# export GARAK_PROBE='latentinjection.LatentInjectionFactSnippetLegal'
# export GARAK_PROBE='latentinjection.LatentInjectionReport'
# export GARAK_PROBE='latentinjection.LatentInjectionResume'
# export GARAK_PROBE='latentinjection.LatentInjectionTranslationEnFr'
# export GARAK_PROBE='latentinjection.LatentInjectionTranslationEnZh'
# export GARAK_PROBE='latentinjection.LatentJailbreak'
# export GARAK_PROBE='latentinjection.LatentWhoisSnippet'
# export GARAK_PROBE='leakreplay.NYTCloze'
# export GARAK_PROBE='leakreplay.NYTComplete'
# export GARAK_PROBE='malwaregen.Payload'
# export GARAK_PROBE='malwaregen.SubFunctions'
# export GARAK_PROBE='malwaregen.TopLevel'
# export GARAK_PROBE='phrasing.FutureTense'
# export GARAK_PROBE='phrasing.PastTense'
export GARAK_PROBE='promptinject.HijackHateHumans'
# export GARAK_PROBE='promptinject.HijackKillHumans'
# export GARAK_PROBE='promptinject.HijackLongPrompt'
# export GARAK_PROBE='snowball.GraphConnectivity'
# export GARAK_PROBE='snowball.Senators'
# export GARAK_PROBE='suffix.BEAST' Microsoft ONNX Phi-3 LLM/GenAI REST API Service does not have both a `model` and `tokenizer` attribute. Cannot run BEAST.
# export GARAK_PROBE='suffix.GCG'
# export GARAK_PROBE='suffix.GCGCached'
# export GARAK_PROBE='tap.TAP'
# export GARAK_PROBE='tap.TAPCached'
# export GARAK_PROBE='topic.WordnetBlockedWords'
# export GARAK_PROBE='topic.WordnetControversial'
# export GARAK_PROBE='visual_jailbreak.FigStep'
# export GARAK_PROBE='xss.MarkdownImageExfil'
# export GARAK_PROBE='xss.MdExfil20230929'
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