from typing import List
from numpy import ndarray


from src.text_generation.domain.abstract_semantic_similarity_result import AbstractSemanticSimilarityResult


class SemanticSimilarityResult(AbstractSemanticSimilarityResult):
    def __init__(self, scores: ndarray):
        super().__init__()
        self.max: float = float(scores.max()) 
        self.mean: float = float(scores.mean())
        self.scores: List[float] = scores.tolist()