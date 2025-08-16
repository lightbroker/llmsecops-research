from src.text_generation.adapters.foundation_models.base.base_model_pipeline import BaseModelPipeline


from typing import Any, Dict, List


class AppleOpenELMPipeline(BaseModelPipeline):
    def get_model_specific_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 256,  # Smaller model, might need fewer tokens
            "temperature": 0.4,     # Override common temperature for this model
            "top_k": 40,           # Add top-k sampling
        }

    def get_stop_sequences(self) -> List[str]:
        return ["</s>", "[/INST]"]