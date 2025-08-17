from typing import Optional
from dataclasses import dataclass
from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig


@dataclass
class AppleOpenELMConfig(BaseModelConfig):
    """OpenELM-specific configuration"""
    use_cache: bool = True
    pad_token_id: Optional[int] = None