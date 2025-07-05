from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.text_generation.common.constants import Constants
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel


class TextGenerationCompletionService(AbstractTextGenerationCompletionService):
    
    def __init__(self, foundation_model: AbstractFoundationModel):
        super().__init__()
        self.language_model_pipeline = foundation_model.create_pipeline()
        self.constants = Constants()

    def _extract_assistant_response(self, text):
        if self.constants.ASSISTANT_TOKEN in text:
            return text.split(self.constants.ASSISTANT_TOKEN)[-1].strip()
        return text

    # TODO - get from config?
    def _get_template(self):
        return """<|system|>
            Act as a conversational agent to respond to the end user's prompt. 
            <|user|>\n
            Question: {question}<|end|>\n
            <|assistant|>
            """

    def create_chain(self):
        template = self._get_template()
        prompt = PromptTemplate.from_template(template)
        return (
            {"question": RunnablePassthrough()}
            | prompt
            | self.language_model_pipeline
            | StrOutputParser()
            | self._extract_assistant_response
        )

    def invoke(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        chain = self.create_chain()
        try:
            response = chain.invoke(user_prompt)
            return response
        except Exception as e:
            raise e