from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, StringPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import FewShotPromptTemplate

from src.text_generation.common.constants import Constants
from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.original_completion_result import OriginalCompletionResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder, AbstractSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


class BaseSecurityGuidelinesService(AbstractSecurityGuidelinesService):
    """Base service for security guidelines implementations."""
    
    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
            config_builder: Optional[AbstractSecurityGuidelinesConfigurationBuilder] = None):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service
        self.llm_configuration_introspection_service = llm_configuration_introspection_service
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
            prompt_template: StringPromptTemplate = self._get_template(user_prompt=user_prompt)
            prompt_value: PromptValue = prompt_template.format_prompt(input=user_prompt)
            prompt_dict = {
                "messages": [
                    {"role": msg.type, "content": msg.content, "additional_kwargs": msg.additional_kwargs}
                    for msg in prompt_value.to_messages()
                ],
                "string_representation": prompt_value.to_string(),
            }

            chain = self._create_chain(prompt_template)
            completion_text=chain.invoke({self.constants.INPUT_VARIABLE_TOKEN: user_prompt})
            llm_config = self.llm_configuration_introspection_service.get_config(chain)

            result = TextGenerationCompletionResult(
                original_result=OriginalCompletionResult(
                    user_prompt=user_prompt,
                    completion_text=completion_text,
                    llm_config=llm_config,
                    full_prompt=prompt_dict
                ),
                guidelines_result=GuidelinesResult(
                    user_prompt=user_prompt,
                    completion_text=completion_text,
                    llm_config=llm_config,
                    full_prompt=prompt_dict
                )
            )
            return result
        except Exception as e:
            raise e