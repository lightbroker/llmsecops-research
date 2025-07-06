from src.text_generation.domain.abstract_guardrail_analyzed_response import AbstractGuardrailAnalyzedResponse


class GuardrailAnalyzedResponse(AbstractGuardrailAnalyzedResponse):
    def __init__(
            self, 
            score: float,
            cosine_similarity_risk_threshold: float,
            original: str):
        is_completion_malicious = score >= cosine_similarity_risk_threshold

        self.score = score
        self.original = original
        self.is_completion_malicious = is_completion_malicious
        self.final = "I can't answer that." if is_completion_malicious else original