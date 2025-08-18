from typing import Dict
from langchain_core.prompts import StringPromptTemplate

from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService

class RagPlusCotSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """
    Service that combines Retrieval Augmented Generation (RAG) with
    Chain of Thought (CoT) security guidelines with dynamic template selection.
    """
    
    def __init__(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: AbstractSecurityGuidelinesConfigurationBuilder):
        
        super().__init__(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )
        
        # Initialize the model-to-rag-plus-cot-template mapping
        self._rag_plus_cot_template_mapping = self._build_rag_plus_cot_template_mapping()

    def _build_rag_plus_cot_template_mapping(self) -> Dict[str, str]:
        """
        Build mapping from model identifiers to their corresponding RAG+CoT template IDs.
        
        Returns:
            Dict[str, str]: Mapping from model name/identifier to RAG+CoT template ID
        """
        return {
            # Phi-3 models
            "phi-3-mini-4k-instruct": self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            "microsoft/phi-3-mini-4k-instruct": self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            
            # OpenELM models
            "openelm-3b-instruct": self.constants.PromptTemplateIds.OPENELM_3B_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            "apple/openelm-3b-instruct": self.constants.PromptTemplateIds.OPENELM_3B_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            
            # Llama models
            "llama-3.2-3b-instruct": self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__04_FEW_SHOT_RAG_PLUS_COT,
            "meta-llama/llama-3.2-3b-instruct": self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__04_FEW_SHOT_RAG_PLUS_COT,
        }

    def _get_model_identifier(self) -> str:
        """
        Get the model identifier from the foundation model.
        
        Returns:
            str: Model identifier/name
        """
        # First try to get from foundation model if available
        if hasattr(self, 'foundation_model') and self.foundation_model:
            model_info = self.foundation_model.get_model_info()
            if model_info:
                model_id = (
                    model_info.get('model_name') or
                    model_info.get('model_id') or
                    model_info.get('name') or
                    str(model_info)
                )
                return model_id.lower() if model_id else ""
        
        # Fallback to introspection service
        try:
            model_info = self.llm_configuration_introspection_service.get_model_configuration()
            
            # Try different possible attribute names for the model identifier
            model_id = (
                getattr(model_info, 'model_name', None) or
                getattr(model_info, 'model_id', None) or
                getattr(model_info, 'name', None) or
                str(model_info)
            )
            
            return model_id.lower() if model_id else ""
        except Exception:
            return ""

    def _get_rag_plus_cot_template_id_for_model(self, model_identifier: str) -> str:
        """
        Get the appropriate RAG+CoT template ID for the given model.
        
        Args:
            model_identifier: The model identifier/name
            
        Returns:
            str: The template ID for RAG+CoT prompting
            
        Raises:
            ValueError: If no RAG+CoT template is found for the model
        """
        # Try exact match first
        if model_identifier in self._rag_plus_cot_template_mapping:
            return self._rag_plus_cot_template_mapping[model_identifier]
        
        # Try partial matches for flexibility
        for model_key, template_id in self._rag_plus_cot_template_mapping.items():
            if model_key in model_identifier or model_identifier in model_key:
                return template_id
        
        # If no match found, raise an informative error
        available_models = list(self._rag_plus_cot_template_mapping.keys())
        raise ValueError(
            f"No RAG+CoT template found for model '{model_identifier}'. "
            f"Available models: {available_models}"
        )

    def get_template(self, user_prompt: str) -> StringPromptTemplate:
        """
        Get RAG+CoT security guidelines template dynamically based on the current model.
        
        Args:
            user_prompt: The user's input prompt
            
        Returns:
            StringPromptTemplate: Template configured for RAG+CoT processing
        """
        # Get the current model identifier
        model_identifier = self._get_model_identifier()
        
        # Get the appropriate RAG+CoT template ID for this model
        template_id = self._get_rag_plus_cot_template_id_for_model(model_identifier)
        
        # Use the config builder to get the template with RAG context
        return self.config_builder.get_prompt_template(
            template_id=template_id,
            user_prompt=user_prompt
        )

    def add_model_template_mapping(self, model_identifier: str, template_id: str) -> None:
        """
        Add or update a model-to-RAG+CoT-template mapping.
        
        Args:
            model_identifier: The model identifier/name
            template_id: The corresponding RAG+CoT template ID
        """
        self._rag_plus_cot_template_mapping[model_identifier.lower()] = template_id

    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model identifiers.
        
        Returns:
            list[str]: List of supported model identifiers
        """
        return list(self._rag_plus_cot_template_mapping.keys())