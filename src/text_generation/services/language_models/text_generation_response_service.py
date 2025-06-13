from src.text_generation.services.language_models.abstract_language_model_response_service import AbstractLanguageModelResponseService
from src.text_generation.adapters.llm.language_model import LanguageModel


class TextGenerationResponseService(AbstractLanguageModelResponseService):
    
    def invoke(self, user_prompt: str) -> str:

        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        llm = LanguageModel()
        response = llm.invoke(user_prompt=user_prompt)
        return response