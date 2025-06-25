import abc


class AbstractGeneratedTextGuardrailService(abc.ABC):
    @abc.abstractmethod
    def analyze(self, model_generated_text: str) -> float:
        raise NotImplementedError