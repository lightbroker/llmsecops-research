from typing import Optional

from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult
from src.text_generation.domain.alternate_completion_result import AlternateCompletionResult
from src.text_generation.domain.guardrails_result import GuardrailsResult
from src.text_generation.domain.guidelines_result import GuidelinesResult


class TextGenerationCompletionResult(
    AbstractTextGenerationCompletionResult):
    """
        Represents the result of a text generation completion 
        with optional security guidelines processing results.
    """
    
    def __init__(
            self, 
            llm_config: dict,
            original_completion: str,
            guidelines_result: Optional[GuidelinesResult] = None,
            guardrails_result: Optional[GuardrailsResult] = None):

        self.llm_config = llm_config
        self.original_completion = original_completion
        self.guidelines_processed_completion = guidelines_result
        self.guardrails_processed_completion = guardrails_result
        self.alternate_result: AlternateCompletionResult = None

        self.final = (
            (self.alternate_result and self.alternate_result.alterate_completion_text) or
            original_completion
        )