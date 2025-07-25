from numpy import array, ndarray
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
        if not comparison_texts:
            raise ValueError('Comparison texts collection cannot be empty')
        self.comparison_texts = comparison_texts

    def analyze(self, text: str) -> AbstractSemanticSimilarityResult:
        """
            Perfect alignment (similarity) results in a score of 1;
            orthogonal is 0; 
            opposite is -1
        """

        if not self.comparison_texts:
            raise ValueError('Comparison texts are required')

        print(f'Using {len(self.comparison_texts)} comparison texts')
        query_embedding = array(self.embeddings.embed_query(text)).reshape(1, -1)
        doc_embeddings = array(self.embeddings.embed_documents(self.comparison_texts))

        similarity_scores: ndarray = cosine_similarity(query_embedding, doc_embeddings)    
        return SemanticSimilarityResult(scores=similarity_scores.flatten())