from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.text_generation.common.constants import Constants
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.prompt_template_service import PromptTemplateService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


class RetrievalAugmentedGenerationContextSecurityGuidelinesService(
    AbstractSecurityGuidelinesService):
    """Implementation of RAG context security guidelines service."""
    def __init__(
            self,
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service: PromptTemplateService = prompt_template_service

    def _create_chain(self, prompt_template: PromptTemplate):
        return (
            { "question": RunnablePassthrough() }
            | prompt_template
            | self.foundation_model_pipeline
            | StrOutputParser()
            | self.response_processing_service.process_text_generation_output
        )

    def apply_guidelines(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        try:
            template_id = self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_FEW_SHOT_EXAMPLES
            prompt_template: PromptTemplate = self.prompt_template_service.get(id=template_id)
            chain = self._create_chain(prompt_template)
            return chain.invoke(user_prompt)
        except Exception as e:
            raise e