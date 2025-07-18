from typing import List
from src.text_generation.ports.abstract_prompt_injection_example_repository import AbstractPromptInjectionExampleRepository
from src.text_generation.services.prompt_injection.abstract_prompt_injection_example_service import AbstractPromptInjectionExampleService

class PromptInjectionExampleService(AbstractPromptInjectionExampleService):
    """Service for handling prompt injection examples."""

    def __init__(self, repository: AbstractPromptInjectionExampleRepository):
        self.repository = repository

    def get_all_prompts(self) -> List[str]:
        """Get all prompt injection prompts."""
        examples = self.repository.get_all()
        return [example["prompt_injection_prompt"] for example in examples]

    def get_all_completions(self) -> List[str]:
        """Get all prompt injection completions."""
        examples = self.repository.get_all()
        return [example["prompt_injection_completion"] for example in examples]