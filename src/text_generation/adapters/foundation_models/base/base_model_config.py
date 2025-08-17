from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseModelConfig:
    """Base configuration for foundation models"""
    max_new_tokens: int = 512
    temperature: float = 0.3
    repetition_penalty: float = 1.1
    use_fast: bool = True
    local_files_only: bool = False
    torch_dtype: str = "auto"
