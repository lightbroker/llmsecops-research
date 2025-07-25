from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, StringPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig

from src.text_generation.common.constants import Constants
from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder, AbstractSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService

class BaseSecurityGuidelinesService(AbstractSecurityGuidelinesService):
    """Base service for security guidelines implementations."""
    
    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            config_builder: Optional[AbstractSecurityGuidelinesConfigurationBuilder] = None):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service
        self.config_builder = config_builder

    def _create_chain(self, prompt_template: PromptTemplate):

        if prompt_template is None:
            raise ValueError("prompt_template cannot be None")

        return (
            { f"{self.constants.INPUT_VARIABLE_TOKEN}": RunnablePassthrough() }
            | prompt_template
            | self.foundation_model_pipeline
            | StrOutputParser()
            | self.response_processing_service.process_text_generation_output
        )

    def _get_template(self, user_prompt: str) -> StringPromptTemplate:
        """
        Get the prompt template for security guidelines.
        
        Returns:
            StringPromptTemplate: Template for processing security guidelines
        """
        raise NotImplementedError("Subclasses must implement _get_template()")

    def apply_guidelines(self, user_prompt: str) -> AbstractGuidelinesProcessedCompletion:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        try:
            prompt_template = self._get_template(user_prompt=user_prompt)
            chain = self._create_chain(prompt_template)
            result = GuidelinesResult(
                completion_text=chain.invoke(user_prompt),
                llm_config=chain.steps[1].model_dump()
            )
            return result
        except Exception as e:
            raise e