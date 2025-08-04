import abc
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def apply_guardrails(self, text_generation_completion_result: AbstractTextGenerationCompletionResult) -> AbstractTextGenerationCompletionResult:
        raise NotImplementedError