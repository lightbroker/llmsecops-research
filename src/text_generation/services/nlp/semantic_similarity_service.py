from numpy import array
from sklearn.metrics.pairwise import cosine_similarity

from src.text_generation.common.constants import Constants
from src.text_generation.domain.abstract_semantic_similarity_result import AbstractSemanticSimilarityResult
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService
from src.text_generation.ports.abstract_embedding_model import AbstractEmbeddingModel


class SemanticSimilarityService(AbstractSemanticSimilarityService):
    def __init__(
            self,
            embedding_model: AbstractEmbeddingModel):
        super().__init__()
        self.embeddings = embedding_model.embeddings
        self.constants = Constants()
    
    def use_comparison_texts(self, comparison_texts: list[str]):
        self.comparison_texts = comparison_texts

    def analyze(self, text: str) -> AbstractSemanticSimilarityResult:
        """
            Perfect alignment (similarity) results in a score of 1; opposite is 0
        """
        query_embedding = array(self.embeddings.embed_query(text)).reshape(1, -1)
        doc_embeddings = array(self.embeddings.embed_documents(self.comparison_texts))

        similarity_scores = cosine_similarity(query_embedding, doc_embeddings)

        all_scores = list()
        for _, score in enumerate(similarity_scores):
            all_scores.append(score)
    
        return SemanticSimilarityResult(scores=all_scores, mean=similarity_scores.mean())