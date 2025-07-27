from typing import Any, List
from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class GuidelinesResult(
    AbstractGuidelinesProcessedCompletion):
    def __init__(
            self,
            user_prompt: str,
            completion_text: str,
            full_prompt: dict[str, Any],
            llm_config: dict,
            cosine_similarity_score: float = 0.0,
            cosine_similarity_risk_threshold: float = 0.0):
        
        self.user_prompt = user_prompt
        self.completion_text = completion_text
        self.full_prompt = full_prompt
        self.llm_config = llm_config
        self.cosine_similarity_score = cosine_similarity_score
        self.cosine_similarity_risk_threshold = cosine_similarity_risk_threshold

    def is_original_completion_malicious(self) -> bool:
        return self.cosine_similarity_score >= self.cosine_similarity_risk_threshold
