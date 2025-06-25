import abc


class SemanticSimilarityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def analyze(self, prompt_input_text: str) -> float:
        raise NotImplementedError