from typing import List
from src.text_generation.domain.abstract_semantic_similarity_result import AbstractSemanticSimilarityResult


class SemanticSimilarityResult(AbstractSemanticSimilarityResult):
    def __init__(self, scores: List[float], mean: float):
        super().__init__()
        self.scores: List[float] = scores
        self.mean: float = mean