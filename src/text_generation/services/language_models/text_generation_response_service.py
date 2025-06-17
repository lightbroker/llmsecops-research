from src.text_generation.services.language_models.abstract_language_model_response_service import AbstractLanguageModelResponseService
from src.text_generation.adapters.llm.language_model import LanguageModel


class TextGenerationResponseService(AbstractLanguageModelResponseService):
    
    def __init__(self, language_model: LanguageModel):
        super().__init__()
        self.language_model = language_model


    def invoke(self, user_prompt: str) -> str:

        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        response = self.language_model.invoke(user_prompt=user_prompt)
        return response