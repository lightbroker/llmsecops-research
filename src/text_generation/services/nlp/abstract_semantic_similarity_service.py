import abc

from src.text_generation.domain.abstract_semantic_similarity_result import AbstractSemanticSimilarityResult


class AbstractSemanticSimilarityService(abc.ABC):
    @abc.abstractmethod
    def analyze(self, text: str) -> AbstractSemanticSimilarityResult:
        raise NotImplementedError
    
    @abc.abstractmethod
    def use_comparison_texts(self, comparison_texts: list[str]):
        raise NotImplementedError