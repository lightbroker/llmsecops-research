from typing import Optional, Dict
from langchain_core.prompts import StringPromptTemplate
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


class ChainOfThoughtSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """Service for zero-shot chain-of-thought security guidelines with dynamic template selection."""
    
    def __init__(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: Optional[AbstractSecurityGuidelinesConfigurationBuilder] = None
    ):
        super().__init__(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )
        
        # Initialize the model-to-template mapping
        self._cot_template_mapping = self._build_cot_template_mapping()
    
    def _build_cot_template_mapping(self) -> Dict[str, str]:
        """
        Build mapping from model identifiers to their corresponding CoT template IDs.
        
        Returns:
            Dict[str, str]: Mapping from model name/identifier to CoT template ID
        """
        return {
            "microsoft/Phi-3-mini-4K-Instruct": self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,   
            "apple/OpenELM-1_1B-Instruct": self.constants.PromptTemplateIds.OPENELM_1_1B_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,
            "meta-llama/Llama-3.2-3B-Instruct": self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,
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
    
    def _get_template(self, user_prompt: str, template_id: str) -> StringPromptTemplate:
        return self.config_builder.get_prompt_template(
            template_id=template_id,
            user_prompt=user_prompt
        )

    def _get_cot_template_id_for_model(self, model_identifier: str) -> str:
        """
        Get the appropriate CoT template ID for the given model.
        
        Args:
            model_identifier: The model identifier/name
            
        Returns:
            str: The template ID for chain of thought prompting
            
        Raises:
            ValueError: If no CoT template is found for the model
        """
        # Try exact match first
        if model_identifier in self._cot_template_mapping:
            return self._cot_template_mapping[model_identifier]
        
        # Try partial matches for flexibility
        for model_key, template_id in self._cot_template_mapping.items():
            if model_key in model_identifier or model_identifier in model_key:
                return template_id
        
        # If no match found, raise an informative error
        available_models = list(self._cot_template_mapping.keys())
        raise ValueError(
            f"No chain of thought template found for model '{model_identifier}'. "
            f"Available models: {available_models}"
        )
    
    def get_template(self, user_prompt: str) -> StringPromptTemplate:
        """
        Get chain of thought security guidelines template dynamically based on the current model.
        
        Args:
            user_prompt: The user's input prompt
            
        Returns:
            StringPromptTemplate: Template configured for CoT processing
        """
        # Get the current model identifier
        model_identifier = self._get_model_identifier()
        
        # Get the appropriate CoT template ID for this model
        template_id = self._get_cot_template_id_for_model(model_identifier)
        
        # Return the template from the service
        return self.prompt_template_service.get(id=template_id)
    
    def add_model_template_mapping(self, model_identifier: str, template_id: str) -> None:
        """
        Add or update a model-to-template mapping.
        
        Args:
            model_identifier: The model identifier/name
            template_id: The corresponding CoT template ID
        """
        self._cot_template_mapping[model_identifier.lower()] = template_id
    
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model identifiers.
        
        Returns:
            list[str]: List of supported model identifiers
        """
        return list(self._cot_template_mapping.keys())