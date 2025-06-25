import abc


class AbstractSemanticSimilarityService(abc.ABC):
    @abc.abstractmethod
    def analyze(self, text: str) -> float:
        raise NotImplementedError
    
    @abc.abstractmethod
    def use_comparison_texts(self, comparison_texts: list[str]):
        raise NotImplementedError