from src.text_generation.services.language_models.abstract_language_model_response_service import AbstractLanguageModelResponseService


class FakeLanguageModelResponseService(AbstractLanguageModelResponseService):
    
    def invoke(self, user_prompt: str) -> str:

        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        return "fake language model response"