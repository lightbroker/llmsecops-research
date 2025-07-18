class Constants:
    PHI_3_ASSISTANT_START_TOKEN = "<|assistant|>"
    PHI_3_END_TOKEN =             "<|end|>"
    PHI_3_SYSTEM_START_TOKEN =    "<|system|>"
    PHI_3_USER_START_TOKEN =      "<|user|>"
    INPUT_VARIABLE_TOKEN =        "input"

    ALT_COMPLETION_TEXT =         "I can't answer that."

    class PromptTemplateIds:
        PHI_3_MINI_4K_INSTRUCT_BASIC =                      "phi-3-mini-4k-instruct-basic"
        PHI_3_MINI_4K_INSTRUCT_ZERO_SHOT_CHAIN_OF_THOUGHT = "phi-3-mini-4k-instruct-zero-shot-cot"
        PHI_3_MINI_4K_INSTRUCT_FEW_SHOT_EXAMPLES =          "phi-3-mini-4k-instruct-few-shot"
        PHI_3_MINI_4K_INSTRUCT_REFLEXION =                  "phi-3-mini-4k-instruct-reflexion"