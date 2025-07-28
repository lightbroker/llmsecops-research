from typing import Any
from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion


class GuardrailsResult:
    """
    Represents the result of guardrails processing without cosine similarity scoring.
    """
    
    def __init__(
        self,
        user_prompt: str,
        completion_text: str,
        full_prompt: dict[str, Any],
        llm_config: dict
    ):
        self.user_prompt = user_prompt
        self.completion_text = completion_text
        self.full_prompt = full_prompt
        self.llm_config = llm_config