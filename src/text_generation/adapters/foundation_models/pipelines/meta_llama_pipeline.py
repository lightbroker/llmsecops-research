from src.text_generation.adapters.foundation_models.base.base_model_pipeline import BaseModelPipeline


from typing import Any, Dict, List


class MetaLlamaPipeline(BaseModelPipeline):
    def get_model_specific_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 512,
            # meta-llama/Llama-3.2-3B-Instruct might need slightly different settings
            "top_p": 0.9,  # Add nucleus sampling for better diversity
        }

    def get_stop_sequences(self) -> List[str]:
        return ["</s>", "[/INST]"]