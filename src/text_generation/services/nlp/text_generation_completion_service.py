from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.text_generation.common.constants import Constants
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel


class TextGenerationCompletionService(AbstractTextGenerationCompletionService):
    
    def __init__(
            self, 
            foundation_model: AbstractFoundationModel,
            prompt_template_service: AbstractPromptTemplateService):
        super().__init__()
        self.language_model_pipeline = foundation_model.create_pipeline()
        self.prompt_template_service = prompt_template_service
        self.constants = Constants()

    def _extract_assistant_response(self, text):
        if self.constants.ASSISTANT_TOKEN in text:
            return text.split(self.constants.ASSISTANT_TOKEN)[-1].strip()
        return text

    def create_chain(self):
        prompt_template_id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_BASIC
        prompt_template = self.prompt_template_service.get(id=prompt_template_id)
        return (
            { "question": RunnablePassthrough() }
            | prompt_template
            | self.language_model_pipeline
            | StrOutputParser()
            | self._extract_assistant_response
        )

    def invoke(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        try:
            chain = self.create_chain()
            return chain.invoke(user_prompt)
        except Exception as e:
            raise e