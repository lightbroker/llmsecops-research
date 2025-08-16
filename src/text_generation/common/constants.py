class Constants:
    # Phi-3 tokens
    PHI_3_ASSISTANT_START_TOKEN = "<|assistant|>"
    PHI_3_END_TOKEN =             "<|end|>"
    PHI_3_SYSTEM_START_TOKEN =    "<|system|>"
    PHI_3_USER_START_TOKEN =      "<|user|>"

    # OpenELM tokens
    OPENELM_SYSTEM_START_TOKEN =  "[INST]"
    OPENELM_USER_START_TOKEN =    "[INST]"
    OPENELM_ASSISTANT_TOKEN =     "[/INST]"
    OPENELM_END_TOKEN =           "</s>"

    # TinyLlama tokens
    TINYLLAMA_SYSTEM_START_TOKEN = "<|system|>"
    TINYLLAMA_USER_START_TOKEN =   "<|user|>"
    TINYLLAMA_ASSISTANT_TOKEN =    "<|assistant|>"
    TINYLLAMA_END_TOKEN =          "</s>"

    INPUT_VARIABLE_TOKEN =        "input"
    ALT_COMPLETION_TEXT =         "I can't answer that."

    class PromptTemplateIds:
        # Phi-3 templates
        PHI_3_MINI_4K_INSTRUCT__01_BASIC =                      "phi-3-mini-4k-instruct.01-basic"
        PHI_3_MINI_4K_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "phi-3-mini-4k-instruct.02-zero-shot-cot"
        PHI_3_MINI_4K_INSTRUCT__03_FEW_SHOT_EXAMPLES =          "phi-3-mini-4k-instruct.03-few-shot"
        PHI_3_MINI_4K_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT =      "phi-3-mini-4k-instruct.04-few-shot-rag-plus-cot"
        PHI_3_MINI_4K_INSTRUCT__05_REFLEXION =                  "phi-3-mini-4k-instruct.05-reflexion"

        # OpenELM templates
        OPENELM_270M_INSTRUCT__01_BASIC =                      "openelm-270m-instruct.01-basic"
        OPENELM_270M_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "openelm-270m-instruct.02-zero-shot-cot"
        OPENELM_270M_INSTRUCT__03_FEW_SHOT_EXAMPLES =          "openelm-270m-instruct.03-few-shot"
        OPENELM_270M_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT =      "openelm-270m-instruct.04-few-shot-rag-plus-cot"
        OPENELM_270M_INSTRUCT__05_REFLEXION =                  "openelm-270m-instruct.05-reflexion"

        # TinyLlama templates
        TINYLLAMA_1_1B_CHAT__01_BASIC =                      "tinyllama-1.1b-chat.01-basic"
        TINYLLAMA_1_1B_CHAT__02_ZERO_SHOT_CHAIN_OF_THOUGHT = "tinyllama-1.1b-chat.02-zero-shot-cot"
        TINYLLAMA_1_1B_CHAT__03_FEW_SHOT_EXAMPLES =          "tinyllama-1.1b-chat.03-few-shot"
        TINYLLAMA_1_1B_CHAT__04_FEW_SHOT_RAG_PLUS_COT =      "tinyllama-1.1b-chat.04-few-shot-rag-plus-cot"
        TINYLLAMA_1_1B_CHAT__05_REFLEXION =                  "tinyllama-1.1b-chat.05-reflexion"