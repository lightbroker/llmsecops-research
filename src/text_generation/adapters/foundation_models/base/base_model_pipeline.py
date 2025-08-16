from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class BaseModelPipeline(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_common_config(self) -> Dict[str, Any]:
        """Common configuration shared across all models"""
        return {
            "do_sample": True,
            "temperature": 0.3,
            "repetition_penalty": 1.1,
            "use_fast": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
    
    @abstractmethod
    def get_model_specific_config(self) -> Dict[str, Any]:
        """Model-specific configuration overrides"""
        pass
    
    @abstractmethod
    def get_stop_sequences(self) -> List[str]:
        """Model-specific stop sequences"""
        pass
    
    def _create_base_pipeline(self):
        """Create the base pipeline with merged configurations"""
        config = self.get_common_config()
        config.update(self.get_model_specific_config())
        
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            **config
        )
    
    def create_pipeline(self) -> HuggingFacePipeline:
        """Create the final HuggingFace pipeline"""
        pipe = self._create_base_pipeline()
        
        return HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "return_full_text": False,
                "stop_sequence": self.get_stop_sequences()
            }
        )



