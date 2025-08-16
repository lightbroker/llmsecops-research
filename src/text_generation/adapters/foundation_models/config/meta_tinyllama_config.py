from dataclasses import dataclass
from typing import Any, Dict, Optional
from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig


@dataclass
class MetaTinyLlamaConfig(BaseModelConfig):
    """TinyLlama-specific configuration"""
    use_flash_attention: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None