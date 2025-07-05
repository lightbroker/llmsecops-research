from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


class FakeTextGenerationCompletionService(AbstractTextGenerationCompletionService):
    
    def invoke(self, user_prompt: str) -> str:

        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        return "fake language model response"