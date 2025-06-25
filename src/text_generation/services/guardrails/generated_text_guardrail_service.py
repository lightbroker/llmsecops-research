from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService


class GeneratedTextGuardrailService(AbstractGeneratedTextGuardrailService):
    def __init__(
            self,
            semantic_similarity_service: AbstractSemanticSimilarityService,
            comparison_texts: list[str]):
        super().__init__()
        self.semantic_similarity_service = semantic_similarity_service
        self.semantic_similarity_service.use_comparison_texts(comparison_texts)
        self.cosine_similarity_risk_threshold: float = 0.5

    def analyze(self, model_generated_text: str) -> float:
        score: float = self.semantic_similarity_service.analyze(text=model_generated_text)
        return score >= self.cosine_similarity_risk_threshold