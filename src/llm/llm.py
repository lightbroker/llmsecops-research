"""
RAG implementation with local Phi-3-mini-4k-instruct-onnx and embeddings
"""

import os
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

# ------------------------------------------------------
# 1. LOAD THE LOCAL PHI-3 MODEL
# ------------------------------------------------------

class Phi3LanguageModel:

    def extract_assistant_response(self, text):
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text


    def invoke(self, user_input: str) -> str:
        # Set up paths to the local model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4")
        print(f"Loading Phi-3 model from: {model_path}")

        # List and print the contents of the model_path directory
        print(f"Listing contents of model directory: {model_path}")
        try:
            files = os.listdir(model_path)
            for i, file in enumerate(files):
                file_path = os.path.join(model_path, file)
                file_size = os.path.getsize(file_path)
                is_dir = os.path.isdir(file_path)
                file_type = "dir" if is_dir else "file"
                print(f"{i+1:2d}. {file:50s} [{file_type}] {file_size:,} bytes")
            print(f"Total: {len(files)} items found")
        except FileNotFoundError:
            print(f"ERROR: Directory {model_path} not found!")
        except PermissionError:
            print(f"ERROR: Permission denied when accessing {model_path}")
        except Exception as e:
            print(f"ERROR: Unexpected error when listing directory: {str(e)}")

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True  # Add this line
        )
        model = ORTModelForCausalLM.from_pretrained(
            model_path,  # Change model_id to just model_path
            provider="CPUExecutionProvider",
            trust_remote_code=True,
            local_files_only=True  # Add this line
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
        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            | self.extract_assistant_response
        )
        
        try:
            # Get response from the chain
            response = chain.invoke(user_input)
            # Print the answer
            print(response)
            return response
        except Exception as e:
            print(f"Failed: {e}")
            return e
        
