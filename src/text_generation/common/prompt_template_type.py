from enum import Enum


class PromptTemplateType(Enum):
    BASIC = "basic"
    ZERO_SHOT_COT = "zero_shot_cot"
    FEW_SHOT = "few_shot"
    RAG_PLUS_COT = "rag_plus_cot"