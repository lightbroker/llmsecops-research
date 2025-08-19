# Factory for creating foundation models
from typing import Optional
from src.text_generation.adapters.foundation_models.apple_openelm_foundation_model import AppleOpenELMFoundationModel
from src.text_generation.adapters.foundation_models.base.base_foundation_model import BaseFoundationModel
from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig
from src.text_generation.adapters.foundation_models.meta_llama_foundation_model import MetaLlamaFoundationModel
from src.text_generation.adapters.foundation_models.microsoft_phi3_foundation_model import MicrosoftPhi3FoundationModel
from src.text_generation.common.model_id import ModelId


class FoundationModelFactory:
    """Factory for creating foundation model instances"""

    @staticmethod
    def create_model(model_id: ModelId, config: Optional[BaseModelConfig] = None) -> BaseFoundationModel:
        if config is None:
            config = BaseModelConfig()

        model_map = {
            ModelId.APPLE_OPENELM_1_1B_INSTRUCT.value: AppleOpenELMFoundationModel,
            ModelId.META_LLAMA_3_2_3B_INSTRUCT.value: MetaLlamaFoundationModel,
            ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT.value: MicrosoftPhi3FoundationModel
        }

        if model_id not in model_map:
            raise ValueError(f"Unsupported model type: {model_id}")

        return model_map[model_id](config)