from numpy import float64, array
from sklearn.metrics.pairwise import cosine_similarity

from src.text_generation.common.constants import Constants
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

    def analyze(self, text: str) -> float:
        query_embedding = self.embeddings.embed_query(text)
        doc_embeddings = self.embeddings.embed_documents(self.comparison_texts)

        query_embedding = array(query_embedding).reshape(1, -1)
        doc_embeddings = array(doc_embeddings)
        similarity_scores: list[float64] = cosine_similarity(query_embedding, doc_embeddings)[0]
        scores = list()

        # perfect alignment (similarity) results in a score of 1;
        # opposite is -1
        for _, score in enumerate(similarity_scores):
            scores.append(score)

        return max(scores)