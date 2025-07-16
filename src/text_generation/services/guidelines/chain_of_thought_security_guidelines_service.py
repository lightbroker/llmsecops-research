from langchain_core.prompts import PromptTemplate

from src.text_generation.common.constants import Constants
from src.text_generation.services.guidelines.abstract_chain_of_thought_security_guidelines_service import AbstractChainOfThoughtSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.prompt_template_service import PromptTemplateService


class ChainOfThoughtSecurityGuidelinesService(
        AbstractChainOfThoughtSecurityGuidelinesService):
    
    def __init__(
            self,
            prompt_template_service: AbstractPromptTemplateService):
        super().__init__()
        self.constants = Constants()
        self.prompt_template_service: PromptTemplateService = prompt_template_service

    def apply_guidelines(self, user_prompt: str) -> str:
        
        template_id = self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_ZERO_SHOT_CHAIN_OF_THOUGHT
        prompt_template: PromptTemplate = self.prompt_template_service.get(id=template_id)
        