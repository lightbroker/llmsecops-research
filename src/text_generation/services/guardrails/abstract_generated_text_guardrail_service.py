import abc
from src.text_generation.domain.abstract_guardrail_processed_completion import AbstractGuardrailProcessedCompletion


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def process_generated_text(self, model_generated_text: str) -> AbstractGuardrailProcessedCompletion:
        raise NotImplementedError