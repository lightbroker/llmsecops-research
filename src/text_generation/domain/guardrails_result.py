from typing import Any
from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult


class GuardrailsResult:
   
    def __init__(
        self,
        user_prompt: str,
        completion_text: str,
        full_prompt: dict[str, Any],
        llm_config: dict,
        cosine_similarity_risk_threshold: float = 0.8
    ):
        self.user_prompt = user_prompt
        self.guardrails_completion_text = completion_text
        self.full_prompt = full_prompt
        self.llm_config = llm_config
        self.cosine_similarity_risk_threshold = cosine_similarity_risk_threshold

    def append_semantic_similarity_result(self, semantic_similarity_result: SemanticSimilarityResult):
        self.semantic_similarity_result = semantic_similarity_result

    def is_completion_malicious(self) -> bool:
        return self.semantic_similarity_result.max >= self.cosine_similarity_risk_threshold