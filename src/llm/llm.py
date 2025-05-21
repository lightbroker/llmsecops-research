"""
RAG implementation with local Phi-3-mini-4k-instruct-onnx and embeddings
"""

import logging
import os
import sys
from typing import List

# LangChain imports
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# HuggingFace and ONNX imports
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline


class Phi3LanguageModel:

    def __init__(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        self.logger = logger
        self.configure_model()

    def configure_model(self):

        # Set up paths to the local model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4")
        self.logger.debug(f"Loading Phi-3 model from: {model_path}")

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = ORTModelForCausalLM.from_pretrained(
            model_path,  # Change model_id to just model_path
            provider="CPUExecutionProvider",
            trust_remote_code=True,
            local_files_only=True
        )
        model.name_or_path = model_path

        # Create the text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_fast=True,
            do_sample=True
        )

        # Create the LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)

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
            self.logger.debug(f'===Prompt: {user_input}\n\n')
            response = self.chain.invoke(user_input)
            # Print the answer
            self.logger.debug(f'===Response: {response}\n\n')
            return response
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            return e
        
