from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.text_generation.adapters.foundation_models.base.base_foundation_model import BaseFoundationModel
from src.text_generation.adapters.foundation_models.config.apple_openelm_config import AppleOpenELMConfig
from src.text_generation.common.model_id import ModelId


class AppleOpenELMFoundationModel(BaseFoundationModel):
    """apple/OpenELM-1_1B-Instruct implementation"""

    MODEL_ID = ModelId.APPLE_OPENELM_1_1B_INSTRUCT
    
    def __init__(self, config: AppleOpenELMConfig = AppleOpenELMConfig()):
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
                "stop_sequence": ["</s>", "[/INST]"]
            })