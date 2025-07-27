from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, StringPromptTemplate
from langchain_core.prompt_values import PromptValue

from src.text_generation.common.constants import Constants
from src.text_generation.domain.guardrails_result import GuardrailsResult
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService

from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService


class ReflexionSecurityGuardrailsService(
    AbstractGeneratedTextGuardrailService):
    """Basic implementation of reflexion security guardrails service."""

    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service
        self.llm_configuration_introspection_service = llm_configuration_introspection_service


    def _create_context_from_rag(self, text_generation_completion_result: AbstractTextGenerationCompletionResult) -> str:
        result: TextGenerationCompletionResult = text_generation_completion_result
        original_user_prompt = result.original_user_prompt
        original_completion = result.original_completion

        # Assemble the context showing the original prompt injection attack
        context_parts = [
            "Original User Prompt:",
            f"'{original_user_prompt}'",
            "",
            "Generated Completion:",
            f"'{original_completion}'"
        ]
        
        return "\n".join(context_parts)

    def _get_template(self, text_generation_completion_result: AbstractTextGenerationCompletionResult) -> StringPromptTemplate:
        prompt_template: StringPromptTemplate = self.prompt_template_service.get(
            id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__05_REFLEXION
        )
        context = self._create_context_from_rag(text_generation_completion_result)
        
        # Create a new template with the context filled in
        filled_template = PromptTemplate(
            input_variables=[self.constants.INPUT_VARIABLE_TOKEN],
            template=prompt_template.template.replace("{context}", context)
        )
        
        return filled_template

    def _create_chain(self, prompt_template: StringPromptTemplate):
        return prompt_template | self.foundation_model_pipeline | StrOutputParser()

    def apply_guardrails(self, text_generation_completion_result: AbstractTextGenerationCompletionResult) -> AbstractTextGenerationCompletionResult:
        """
        Apply reflexion-based guardrails to mitigate prompt injection attacks
        """
        if not text_generation_completion_result:
            raise ValueError(f"Parameter 'text_generation_completion_result' cannot be empty or None")
        
        try:
            result: TextGenerationCompletionResult = text_generation_completion_result
            original_user_prompt = result.original_user_prompt
            
            prompt_template: StringPromptTemplate = self._get_template(text_generation_completion_result)
            prompt_value: PromptValue = prompt_template.format_prompt(**{self.constants.INPUT_VARIABLE_TOKEN: original_user_prompt})
            
            prompt_dict = {
                "messages": [
                    {"role": msg.type, "content": msg.content, "additional_kwargs": msg.additional_kwargs}
                    for msg in prompt_value.to_messages()
                ],
                "string_representation": prompt_value.to_string(),
            }

            chain = self._create_chain(prompt_template)
            completion_text = chain.invoke({self.constants.INPUT_VARIABLE_TOKEN: original_user_prompt})
            llm_config = self.llm_configuration_introspection_service.get_config(chain)
            
            result.guardrails_processed_completion = GuardrailsResult(
                completion_text=completion_text,
                llm_config=llm_config,
                full_prompt=prompt_dict
            )
            return result
            
        except Exception as e:
            raise e