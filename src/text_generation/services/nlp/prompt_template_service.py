from langchain_core.prompts import PromptTemplate, StringPromptTemplate
from src.text_generation.ports.abstract_prompt_template_repository import AbstractPromptTemplateRepository
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService


class PromptTemplateService(AbstractPromptTemplateService):
    def __init__(
            self, 
            prompt_template_repository: AbstractPromptTemplateRepository):
        super().__init__()
        self.prompt_template_repository = prompt_template_repository

    def get(self, id: str) -> StringPromptTemplate:
        prompt_template: StringPromptTemplate = self.prompt_template_repository.get(id)
        return prompt_template
    
    def add(self, id: str, prompt_template: PromptTemplate) -> None:
        self.prompt_template_repository.add(id, prompt_template)