import abc


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def is_text_malicious(self, model_generated_text: str) -> float:
        raise NotImplementedError