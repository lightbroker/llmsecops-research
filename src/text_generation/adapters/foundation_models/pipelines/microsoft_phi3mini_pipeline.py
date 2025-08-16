from src.text_generation.adapters.foundation_models.base.base_model_pipeline import BaseModelPipeline


from typing import Any, Dict, List


class MicrosoftPhi3MiniPipeline(BaseModelPipeline):
    def get_model_specific_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 512,
            # Remove max_length to fix the warning - max_new_tokens takes precedence
        }

    def get_stop_sequences(self) -> List[str]:
        return ["<|end|>", "<|user|>", "</s>"]