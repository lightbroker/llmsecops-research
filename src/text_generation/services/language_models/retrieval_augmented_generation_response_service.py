from src.text_generation.adapters.llm.embedding_model import EmbeddingModel
from src.text_generation.adapters.llm.language_model_with_rag import LanguageModelWithRag
from src.text_generation.services.language_models.abstract_language_model_response_service import AbstractLanguageModelResponseService


class RetrievalAugmentedGenerationResponseService(AbstractLanguageModelResponseService):

    def __init__(self, embedding_model: EmbeddingModel):
        super().__init__()
        self.embeddings = embedding_model.embeddings
        self.rag = LanguageModelWithRag(embeddings=self.embeddings)

    def invoke(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")

        response = self.rag.invoke(user_prompt=user_prompt)
        return response