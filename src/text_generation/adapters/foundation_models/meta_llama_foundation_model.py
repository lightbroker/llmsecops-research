import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.text_generation.adapters.foundation_models.config.meta_llama_config import MetaLlamaConfig
from src.text_generation.adapters.foundation_models.base.base_foundation_model import BaseFoundationModel
from src.text_generation.common.model_id import ModelId
from huggingface_hub import login

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
else:
    raise ValueError("HF_TOKEN environment variable not found")


class MetaLlamaFoundationModel(BaseFoundationModel):
    """meta-llama/Llama-3.2-3B-Instruct implementation"""

    MODEL_ID = ModelId.META_LLAMA_3_2_3B_INSTRUCT
    
    def __init__(self, config: MetaLlamaConfig = MetaLlamaConfig()):
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