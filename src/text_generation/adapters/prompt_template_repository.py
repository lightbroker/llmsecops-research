import os
from src.text_generation.ports.abstract_prompt_template_repository import AbstractPromptTemplateRepository
from langchain_core.prompts import load_prompt, PromptTemplate


class PromptTemplateRepository(AbstractPromptTemplateRepository):
    def __init__(self):
        super().__init__()
        self.templates_dir = os.environ.get('PROMPT_TEMPLATES_DIR')
    
    def _create_path_from_id(self, id: str) -> str:
        template_filename = f'{id}.json'
        return os.path.join(self.templates_dir, template_filename)

    def get(self, id: str) -> PromptTemplate:
        return load_prompt(self._create_path_from_id(id))
    
    def add(self, id: str, prompt_template: PromptTemplate) -> None:
        prompt_template.save(self._create_path_from_id(id))