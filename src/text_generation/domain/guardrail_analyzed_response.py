class GuardrailAnalyzedResponse:
        def __init__(
                    self, 
                    score: float,
                    cosine_similarity_risk_threshold: float,
                    original: str,
                    final: str):
            self.score = score
            self.is_malicious = score >= cosine_similarity_risk_threshold
            self.original = original
            self.final = final