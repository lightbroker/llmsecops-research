from langchain_huggingface import HuggingFaceEmbeddings
from src.text_generation.adapters.llm.abstract_embedding_model import AbstractEmbeddingModel


class EmbeddingModel(AbstractEmbeddingModel):

    @property
    def embeddings(self):
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )