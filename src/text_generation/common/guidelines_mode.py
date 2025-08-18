from enum import Enum

class GuidelinesMode(Enum):
    """Enum to define different guidelines processing modes"""
    RAG_PLUS_COT = "rag_plus_cot"
    COT_ONLY = "cot_only"
    RAG_ONLY = "rag_only"
    NONE = "none"