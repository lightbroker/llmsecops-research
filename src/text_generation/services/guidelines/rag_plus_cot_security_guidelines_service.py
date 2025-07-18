from langchain_core.prompts import StringPromptTemplate

from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService

class RagPlusCotSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """
    Service that combines Retrieval Augmented Generation (RAG) with 
    Chain of Thought (CoT) security guidelines.
    """

    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            config_builder: AbstractSecurityGuidelinesConfigurationBuilder):
        super().__init__(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            config_builder=config_builder
        )

    def _get_template(self, user_prompt: str) -> StringPromptTemplate:
        """
        Get RAG context security guidelines template.
        
        Returns:
            StringPromptTemplate: Template configured for RAG processing
        """
        return self.prompt_template_service.get(
            id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_FEW_SHOT_RAG_PLUS_COT
        )

    def _get_template_id(self) -> str:
        """
        Get template ID for combined RAG + CoT processing.
        
        Returns:
            str: Template ID for RAG + CoT security guidelines
        """
        return self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_FEW_SHOT_RAG_PLUS_COT