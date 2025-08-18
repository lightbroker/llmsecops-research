class Constants:
    # Phi-3 tokens
    PHI_3_ASSISTANT_START_TOKEN = "<|assistant|>"
    PHI_3_END_TOKEN =             "<|end|>"
    PHI_3_SYSTEM_START_TOKEN =    "<|system|>"
    PHI_3_USER_START_TOKEN =      "<|user|>"

    # OpenELM tokens
    OPENELM_END_TOKEN =           "</s>"

    # meta-llama/Llama-3.2-3B-Instruct tokens
    LLAMA_SYSTEM_START_TOKEN = "### SYSTEM:\n"
    LLAMA_USER_START_TOKEN =   "### HUMAN:\n"
    LLAMA_ASSISTANT_TOKEN =    "### ASSISTANT:\n"
    LLAMA_END_TOKEN =          ""

    INPUT_VARIABLE_TOKEN =        "input"
    ALT_COMPLETION_TEXT =         "I can't answer that."

    class PromptTemplateIds:
        # Phi-3 templates
        PHI_3_MINI_4K_INSTRUCT__01_BASIC =                      "phi-3-mini-4k-instruct.01-basic"
        PHI_3_MINI_4K_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "phi-3-mini-4k-instruct.02-zero-shot-cot"
        PHI_3_MINI_4K_INSTRUCT__03_FEW_SHOT_EXAMPLES =          "phi-3-mini-4k-instruct.03-few-shot"
        PHI_3_MINI_4K_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT =      "phi-3-mini-4k-instruct.04-few-shot-rag-plus-cot"

        # OpenELM templates: apple/OpenELM-3B-Instruct
        OPENELM_3B_INSTRUCT__01_BASIC =                      "openelm-3b-instruct.01-basic"
        OPENELM_3B_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "openelm-3b-instruct.02-zero-shot-cot"
        OPENELM_3B_INSTRUCT__03_FEW_SHOT_EXAMPLES =          "openelm-3b-instruct.03-few-shot"
        OPENELM_3B_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT =      "openelm-3b-instruct.04-few-shot-rag-plus-cot"

        # meta-llama/Llama-3.2-3B-Instruct templates
        LLAMA_1_1B_CHAT__01_BASIC =                      "llama-3.2-3b-instruct.01-basic"
        LLAMA_1_1B_CHAT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "llama-3.2-3b-instruct.02-zero-shot-cot"
        LLAMA_1_1B_CHAT__03_FEW_SHOT_EXAMPLES =          "llama-3.2-3b-instruct.03-few-shot"
        LLAMA_1_1B_CHAT__04_FEW_SHOT_RAG_PLUS_COT =      "llama-3.2-3b-instruct.04-few-shot-rag-plus-cot"