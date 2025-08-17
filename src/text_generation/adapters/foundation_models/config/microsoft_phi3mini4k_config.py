from dataclasses import dataclass
from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig


@dataclass
class MicrosoftPhi3Mini4KConfig(BaseModelConfig):
    """Phi3-specific configuration"""
    trust_remote_code: bool = True
    torch_dtype: str = "auto"