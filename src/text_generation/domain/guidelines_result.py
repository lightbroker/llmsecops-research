from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class GuidelinesResult(
    AbstractGuidelinesProcessedCompletion):
    def __init__(
            self,
            cosine_similarity_score: float,
            cosine_similarity_risk_threshold: float,
            original_completion: str):
        
        self.score = cosine_similarity_score
        self.original_completion = original_completion
        self.is_original_completion_malicious = (
            cosine_similarity_score >= cosine_similarity_risk_threshold
        )