from typing import List
from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class GuidelinesResult(
    AbstractGuidelinesProcessedCompletion):
    def __init__(
            self,
            completion_text: str,
            llm_config: dict,
            cosine_similarity_score: float,
            cosine_similarity_risk_threshold: float):
        
        self.completion_text = completion_text
        self.llm_config = llm_config
        self.cosine_similarity_score = cosine_similarity_score
        self.cosine_similarity_risk_threshold = cosine_similarity_risk_threshold