import numpy
from sklearn.metrics.pairwise import cosine_similarity

from src.text_generation.adapters.llm.abstract_embedding_model import AbstractEmbeddingModel
from src.text_generation.services.similarity_scoring.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService


class GeneratedTextGuardrailService(AbstractGeneratedTextGuardrailService):
    def __init__(
            self, 
            embedding_model: AbstractEmbeddingModel,
            comparison_texts: list[str]):
        super().__init__()
        self.embeddings = embedding_model.embeddings
        self.comparison_texts = comparison_texts

    def analyze(self, model_generated_text: str) -> float:
        # Get embeddings
        query_embedding = self.embeddings.embed_query(model_generated_text)
        doc_embeddings = self.embeddings.embed_documents(self.comparison_texts)

        # Calculate similarity scores
        query_embedding = numpy.array(query_embedding).reshape(1, -1)
        doc_embeddings = numpy.array(doc_embeddings)

        similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        scores = list()

        # Results will be floating point values between -1 and 1
        for i, score in enumerate(similarity_scores):
            print(f"========  Text {i+1}: {score:.4f} | Score type: {type(score)}")
            scores.append(score)

        return max(scores)