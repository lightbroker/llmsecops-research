# Factory for creating foundation models
from typing import Optional

from langchain_huggingface import HuggingFacePipeline
from src.text_generation.adapters.foundation_models.apple_openelm_foundation_model import AppleOpenELMFoundationModel
from src.text_generation.adapters.foundation_models.base.base_foundation_model import BaseFoundationModel
from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig
from src.text_generation.adapters.foundation_models.meta_tinyllama_foundation_model import MetaTinyLlamaFoundationModel
from src.text_generation.adapters.foundation_models.microsoft_phi3_foundation_model import MicrosoftPhi3FoundationModel
from src.text_generation.adapters.foundation_models.pipelines.apple_openelm_pipeline import AppleOpenELMPipeline
from src.text_generation.adapters.foundation_models.pipelines.meta_tinyllama_pipeline import MetaTinyLlamaPipeline
from src.text_generation.adapters.foundation_models.pipelines.microsoft_phi3mini_pipeline import MicrosoftPhi3MiniPipeline
from src.text_generation.common.model_id import ModelId


class FoundationModelFactory:
    """Factory for creating foundation model instances"""

    @staticmethod
    def create_model(model_id: ModelId, config: Optional[BaseModelConfig] = None) -> BaseFoundationModel:
        if config is None:
            config = BaseModelConfig()

        model_map = {
            ModelId.APPLE_OPENELM_270M_INSTRUCT.value: AppleOpenELMFoundationModel,
            ModelId.META_TINYLLAMA_1_1B_CHAT.value: MetaTinyLlamaFoundationModel,
            ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT.value: MicrosoftPhi3FoundationModel
        }

        if model_id not in model_map:
            raise ValueError(f"Unsupported model type: {model_id}")

        return model_map[model_id](config)
    
    # Factory function to create the appropriate pipeline
    def create_model_pipeline(model_id: ModelId, model, tokenizer) -> HuggingFacePipeline:
        """Factory function to create the appropriate pipeline based on model name"""
        
        pipeline_classes = {
            ModelId.APPLE_OPENELM_270M_INSTRUCT.value: AppleOpenELMPipeline,
            ModelId.META_TINYLLAMA_1_1B_CHAT.value: MetaTinyLlamaPipeline,
            ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT.value: MicrosoftPhi3MiniPipeline
        }
        
        # Determine model type from name
        model_type = None
        for key in pipeline_classes.keys():
            if key in model_id:
                model_type = key
                break
        
        if model_type is None:
            raise ValueError(f"Unsupported model: {model_id}")
        
        pipeline_class = pipeline_classes[model_type]
        return pipeline_class(model, tokenizer).create_pipeline()