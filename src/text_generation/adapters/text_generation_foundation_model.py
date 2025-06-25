import os

from langchain_huggingface import HuggingFacePipeline
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel



class TextGenerationFoundationModel(AbstractFoundationModel):

    def __init__(self):
        model_base_dir = os.environ.get('MODEL_BASE_DIR')
        model_cpu_dir = os.environ.get('MODEL_CPU_DIR')
        model_path = os.path.join(model_base_dir, model_cpu_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = ORTModelForCausalLM.from_pretrained(
            model_path, 
            provider="CPUExecutionProvider",
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.name_or_path = model_path

    
    def create_pipeline(self) -> HuggingFacePipeline:

        pipe = pipeline(
            "text-generation",
            do_sample=True,
            max_new_tokens=512,
            model=self.model,
            repetition_penalty=1.1,
            temperature=0.3,
            tokenizer=self.tokenizer,
            use_fast=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "return_full_text": False,
                "stop_sequence": ["<|end|>", "<|user|>", "</s>"]
            })

