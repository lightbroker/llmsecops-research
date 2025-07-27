from typing import Optional
from langchain_core.prompts import StringPromptTemplate

from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService

class ChainOfThoughtSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """Service for zero-shot chain-of-thought security guidelines."""

    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
            config_builder: Optional[AbstractSecurityGuidelinesConfigurationBuilder] = None):
        super().__init__(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )

    def _get_template(self, user_prompt: str) -> StringPromptTemplate:
        """
        Get chain of thought security guidelines template.
        
        Returns:
            StringPromptTemplate: Template configured for CoT processing
        """
        return self.prompt_template_service.get(
            id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT
        )