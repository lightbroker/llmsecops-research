from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion


class GuardrailsResult(
    AbstractGuardrailsProcessedCompletion):
    def __init__(
            self, 
            cosine_similarity_score: float,
            cosine_similarity_risk_threshold: float,
            original_completion: str,
            guardrails_processed_completion_text: str):
        is_original_completion_malicious = cosine_similarity_score >= cosine_similarity_risk_threshold

        self.score = cosine_similarity_score
        self.original_completion = original_completion
        self.is_original_completion_malicious = is_original_completion_malicious
        self.guardrails_processed_completion_text = guardrails_processed_completion_text