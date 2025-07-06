from src.text_generation.domain.guardrail_analyzed_response import GuardrailAnalyzedResponse
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

    def is_text_malicious(self, model_generated_text: str) -> GuardrailAnalyzedResponse:
        score: float = self.semantic_similarity_service.analyze(text=model_generated_text)
        response = GuardrailAnalyzedResponse(
            score=score,
            cosine_similarity_risk_threshold=self.cosine_similarity_risk_threshold,
            original=model_generated_text,
            final="test")
        return response

