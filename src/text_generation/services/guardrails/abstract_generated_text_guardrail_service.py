import abc
from src.text_generation.domain.abstract_guardrail_analyzed_response import AbstractGuardrailAnalyzedResponse


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def process_generated_text(self, model_generated_text: str) -> AbstractGuardrailAnalyzedResponse:
        raise NotImplementedError