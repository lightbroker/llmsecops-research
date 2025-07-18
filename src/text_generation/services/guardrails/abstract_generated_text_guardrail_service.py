import abc
from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def process_generated_text(self, model_generated_text: AbstractTextGenerationCompletionResult) -> AbstractGuardrailsProcessedCompletion:
        raise NotImplementedError