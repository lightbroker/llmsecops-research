from src.text_generation.domain.abstract_guardrail_processed_completion import AbstractGuardrailProcessedCompletion
from src.text_generation.domain.guardrail_processed_completion import GuardrailProcessedCompletion
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

    def process_generated_text(self, model_generated_text: str) -> AbstractGuardrailProcessedCompletion:
        score: float = self.semantic_similarity_service.analyze(text=model_generated_text)
        response = GuardrailProcessedCompletion(
            score=score,
            cosine_similarity_risk_threshold=self.cosine_similarity_risk_threshold,
            original_completion=model_generated_text)
        return response