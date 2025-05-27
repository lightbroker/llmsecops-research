"""
RAG implementation with local Phi-3-mini-4k-instruct-onnx and embeddings
"""

import logging
import sys

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.text_generation.adapters.llm.text_generation_model import TextGenerationFoundationModel


class Phi3LanguageModel:

    def __init__(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        self.logger = logger
        self.configure_model()

    def configure_model(self):

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
            | self.extract_assistant_response
        )

    def extract_assistant_response(self, text):
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text


    def invoke(self, user_input: str) -> str:
        try:
            # Get response from the chain
            response = self.chain.invoke(user_input)
            return response
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            return e
        
