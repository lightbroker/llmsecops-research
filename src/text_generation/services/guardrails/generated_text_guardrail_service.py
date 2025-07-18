from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult
from src.text_generation.domain.guardrails_result import GuardrailsResult
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService


class GeneratedTextGuardrailService(
    AbstractGeneratedTextGuardrailService):
    def __init__(
            self,
            semantic_similarity_service: AbstractSemanticSimilarityService):
        super().__init__()
        self.semantic_similarity_service = semantic_similarity_service
        self.cosine_similarity_risk_threshold: float = 0.5

    def use_comparison_texts(self, comparison_texts: list[str]):
        self.semantic_similarity_service.use_comparison_texts(comparison_texts)

    def process_generated_text(self, completion: AbstractTextGenerationCompletionResult) -> AbstractGuardrailsProcessedCompletion:
        score: float = self.semantic_similarity_service.analyze(text=completion)
        response = GuardrailsResult(
            cosine_similarity_score=score,
            cosine_similarity_risk_threshold=self.cosine_similarity_risk_threshold,
            original_completion=completion)
        return response