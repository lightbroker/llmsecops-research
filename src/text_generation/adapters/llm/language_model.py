import logging
import sys

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.text_generation.adapters.llm.abstract_language_model import AbstractLanguageModel
from src.text_generation.adapters.llm.text_generation_foundation_model import TextGenerationFoundationModel


class LanguageModel(AbstractLanguageModel):

    def __init__(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        self.logger = logger
        self._configure_model()

    def _extract_assistant_response(self, text):
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text

    def _configure_model(self):

        # Create the LangChain LLM
        llm = TextGenerationFoundationModel().build()

        # Phi-3 specific prompt template
        template = """<|user|>
        Act as a conversational agent to respond to the end user's prompt. 
        Question: {question}
        <|assistant|>
        """

        prompt = PromptTemplate.from_template(template)
        
        # Create a chain using LCEL
        self.chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            | self._extract_assistant_response
        )

    def invoke(self, user_prompt: str) -> str:
        try:
            # Get response from the chain
            response = self.chain.invoke(user_prompt)
            return response
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            raise e
        
