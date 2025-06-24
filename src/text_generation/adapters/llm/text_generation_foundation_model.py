import logging
import os
import sys

from langchain_huggingface import HuggingFacePipeline
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline


class TextGenerationFoundationModel:

    def __init__(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        self.logger = logger

    def build(self) -> HuggingFacePipeline:

        # Set up paths to the local model
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(base_dir, "cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4")
        
        model_base_dir = os.environ.get('MODEL_BASE_DIR')
        model_cpu_dir = os.environ.get('MODEL_CPU_DIR')
        model_path = os.path.join(model_base_dir, model_cpu_dir)
        
        self.logger.debug(f'model_base_dir: {model_base_dir}')
        self.logger.debug(f'model_cpu_dir: {model_cpu_dir}')
        self.logger.debug(f'Loading Phi-3 model from: {model_path}')

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = ORTModelForCausalLM.from_pretrained(
            model_path, 
            provider="CPUExecutionProvider",
            trust_remote_code=True,
            local_files_only=True
        )
        model.name_or_path = model_path

        # Create the text generation pipeline
        pipe = pipeline(
            "text-generation",
            do_sample=True,
            max_new_tokens=512,
            model=model,
            repetition_penalty=1.1,
            temperature=0.3,
            tokenizer=tokenizer,
            use_fast=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Create the LangChain LLM
        return HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "return_full_text": False,
                "stop_sequence": ["<|end|>", "<|user|>", "</s>"]
            })

