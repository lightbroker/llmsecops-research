from typing import Any
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class OriginalCompletionResult(AbstractTextGenerationCompletionResult):
    """
    Represents the original completion result with its own cosine similarity scoring.
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
        self.cosine_similarity_score: float = -1.0
        self.cosine_similarity_risk_threshold: float = 0.0

    def is_completion_malicious(self) -> bool:
        return self.cosine_similarity_score >= self.cosine_similarity_risk_threshold