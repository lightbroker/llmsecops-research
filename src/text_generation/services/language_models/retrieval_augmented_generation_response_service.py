from src.text_generation.adapters.llm.language_model_with_rag import LanguageModelWithRag
from src.text_generation.services.language_models.abstract_language_model_response_service import AbstractLanguageModelResponseService


class RetrievalAugmentedGenerationResponseService(AbstractLanguageModelResponseService):

    def invoke(self, user_prompt: str) -> str:

        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")

        rag = LanguageModelWithRag()
        response = rag.invoke(user_prompt=user_prompt)
        return response