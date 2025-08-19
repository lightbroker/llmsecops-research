from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.text_generation.adapters.foundation_models.config.microsoft_phi3mini4k_config import MicrosoftPhi3Mini4KConfig
from src.text_generation.adapters.foundation_models.base.base_foundation_model import BaseFoundationModel
from src.text_generation.common.model_id import ModelId


class MicrosoftPhi3FoundationModel(BaseFoundationModel):
    """Microsoft Phi3 Mini 4K implementation"""
    
    MODEL_ID = ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT
    
    def __init__(self, config: MicrosoftPhi3Mini4KConfig = MicrosoftPhi3Mini4KConfig()):
        self.config = config
        super().__init__(config)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID.value,
            local_files_only=self.config.local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID.value,
            local_files_only=self.config.local_files_only
        )

    def create_pipeline(self) -> HuggingFacePipeline:
        pipe = self._create_base_pipeline()
        return HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "return_full_text": False,
                "stop_sequence": ["<|end|>", "<|user|>", "</s>"]
            })