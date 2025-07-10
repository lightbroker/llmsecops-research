from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class GuidelinesProcessedCompletion(
    AbstractGuidelinesProcessedCompletion):
    def __init__(
            self, 
            score: float,
            cosine_similarity_risk_threshold: float,
            original_completion: str,
            final: str):
        is_original_completion_malicious = score >= cosine_similarity_risk_threshold

        self.score = score
        self.original_completion = original_completion
        self.is_original_completion_malicious = is_original_completion_malicious
        self.final = final