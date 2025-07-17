from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.domain.guardrails_processed_completion import GuardrailsProcessedCompletion
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService


class GeneratedTextGuardrailService(AbstractGeneratedTextGuardrailService):
    def __init__(
            self,
            semantic_similarity_service: AbstractSemanticSimilarityService):
        super().__init__()
        self.semantic_similarity_service = semantic_similarity_service
        self.cosine_similarity_risk_threshold: float = 0.5

    def use_comparison_texts(self, comparison_texts: list[str]):
        self.semantic_similarity_service.use_comparison_texts(comparison_texts)

    def process_generated_text(self, model_generated_text: str) -> AbstractGuardrailsProcessedCompletion:
        score: float = self.semantic_similarity_service.analyze(text=model_generated_text)
        response = GuardrailsProcessedCompletion(
            score=score,
            cosine_similarity_risk_threshold=self.cosine_similarity_risk_threshold,
            original_completion=model_generated_text)
        return response