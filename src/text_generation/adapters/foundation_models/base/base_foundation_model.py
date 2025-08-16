from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel


from transformers import pipeline


from abc import abstractmethod
from typing import Any


class BaseFoundationModel(AbstractFoundationModel):
    """Base class for all foundation models"""

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model implementation"""
        pass

    def _create_base_pipeline(self) -> Any:
        """Create common pipeline configuration"""
        return pipeline(
            "text-generation",
            do_sample=True,
            max_new_tokens=self.config.max_new_tokens,
            model=self.model,
            repetition_penalty=self.config.repetition_penalty,
            temperature=self.config.temperature,
            tokenizer=self.tokenizer,
            use_fast=self.config.use_fast,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )