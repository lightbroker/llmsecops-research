from enum import Enum
from typing import Optional, Dict, Any
import logging

from langchain_core.prompts import StringPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompt_values import PromptValue

from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig
from src.text_generation.adapters.foundation_models.factories.foundation_model_factory import FoundationModelFactory
from src.text_generation.common.constants import Constants
from src.text_generation.common.guidelines_mode import GuidelinesMode
from src.text_generation.common.model_id import ModelId
from src.text_generation.common.prompt_template_type import PromptTemplateType
from src.text_generation.domain.alternate_completion_result import AlternateCompletionResult
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.original_completion_result import OriginalCompletionResult
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder, AbstractSecurityGuidelinesService
from src.text_generation.services.guidelines.guidelines_factory import AbstractGuidelinesFactory
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.prompt_injection.abstract_prompt_injection_example_service import AbstractPromptInjectionExampleService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


logger = logging.getLogger(__name__)

class TextGenerationCompletionService(AbstractTextGenerationCompletionService):
    def __init__(
            self, 
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            guidelines_factory: AbstractGuidelinesFactory,
            guidelines_config_builder: AbstractSecurityGuidelinesConfigurationBuilder,
            semantic_similarity_service: AbstractSemanticSimilarityService,
            prompt_injection_example_service: AbstractPromptInjectionExampleService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
            default_model_type: ModelId = ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT
    ):
        
        super().__init__()
        self.constants = Constants()
        
        # Model management
        self._current_model = None
        self._current_model_id = None
        self.default_model_id = default_model_type
        self.factory = FoundationModelFactory()
        
        # Services
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service
        self.semantic_similarity_service = semantic_similarity_service
        self.llm_configuration_introspection_service = llm_configuration_introspection_service

        # Set up semantic similarity service
        self.example_prompt_injection_completions = prompt_injection_example_service.get_all_completions()
        self.example_prompt_injection_prompts = prompt_injection_example_service.get_all_prompts()
        self.semantic_similarity_service.use_comparison_texts(
            self.example_prompt_injection_completions
        )

        # Guidelines services
        self.guidelines_factory = guidelines_factory
        self.guidelines_config_builder = guidelines_config_builder

        # Constants and settings
        self.COSINE_SIMILARITY_RISK_THRESHOLD = 0.8
        self._use_guidelines = False
        self._use_zero_shot_chain_of_thought = False
        self._use_rag_context = False

        # Strategy map for guidelines
        self.guidelines_strategy_map = {
            (True, True):   self._handle_cot_and_rag,
            (True, False):  self._handle_cot_only,
            (False, True):  self._handle_rag_only,
            (False, False): self._handle_without_guidelines,
        }

        # Load default model
        self.load_model(default_model_type)

    def _prompt_template_map(self) -> Dict[str, Dict[str, str]]:
        """
        Build mapping from model identifiers to their corresponding template IDs for all template types.
        
        Returns:
            Dict[str, Dict[str, str]]: Mapping from model name/identifier to all template IDs
        """
        return {
            # Phi-3 models
            "microsoft/phi-3-mini-4k-instruct": {
                PromptTemplateType.BASIC.value: self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__01_BASIC,
                PromptTemplateType.ZERO_SHOT_COT.value: self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,
                PromptTemplateType.FEW_SHOT.value: self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__03_FEW_SHOT_EXAMPLES,
                PromptTemplateType.RAG_PLUS_COT.value: self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            },

            # OpenELM models
            "apple/OpenELM-1_1B-Instruct": {
                PromptTemplateType.BASIC.value: self.constants.PromptTemplateIds.OPENELM_1_1B_INSTRUCT__01_BASIC,
                PromptTemplateType.ZERO_SHOT_COT.value: self.constants.PromptTemplateIds.OPENELM_1_1B_INSTRUCT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,
                PromptTemplateType.FEW_SHOT.value: self.constants.PromptTemplateIds.OPENELM_1_1B_INSTRUCT__03_FEW_SHOT_EXAMPLES,
                PromptTemplateType.RAG_PLUS_COT.value: self.constants.PromptTemplateIds.OPENELM_1_1B_INSTRUCT__04_FEW_SHOT_RAG_PLUS_COT,
            },

            # Llama models
            "meta-llama/llama-3.2-3b-instruct": {
                PromptTemplateType.BASIC.value: self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__01_BASIC,
                PromptTemplateType.ZERO_SHOT_COT.value: self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__02_ZERO_SHOT_CHAIN_OF_THOUGHT,
                PromptTemplateType.FEW_SHOT.value: self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__03_FEW_SHOT_EXAMPLES,
                PromptTemplateType.RAG_PLUS_COT.value: self.constants.PromptTemplateIds.LLAMA_1_1B_CHAT__04_FEW_SHOT_RAG_PLUS_COT,
            }
        }

    def _get_model_identifier_from_model_id(self, model_id: ModelId) -> str:
        """
        Get model identifier string from ModelId enum.
        
        Args:
            model_id: The ModelId enum value
            
        Returns:
            str: Model identifier string in lowercase
        """
        # Extract the model name from the enum value
        model_name = model_id.value.lower()
        return model_name

    def _get_current_model_identifier(self) -> str:
        """
        Get the current model identifier.
        
        Returns:
            str: Current model identifier
        """
        if self._current_model_id:
            return self._get_model_identifier_from_model_id(self._current_model_id)
        
        # Fallback: try to get from the actual model instance
        if self._current_model and hasattr(self._current_model, 'get_model_info'):
            model_info = self._current_model.get_model_info()
            if model_info:
                return str(model_info.get('model_name', '')).lower()
        
        return ""

    def load_model(
        self, 
        model_id: ModelId, 
        config: Optional[BaseModelConfig] = None,
        force_reload: bool = False
    ) -> None:
        """Load a specific model"""
        if (not force_reload and 
            self._current_model is not None and 
            self._current_model_id == model_id
        ):
            logger.info(f"Model {model_id.value} already loaded")
            return
        
        self._current_model = self.factory.create_model(model_id, config)
        self._current_model_id: ModelId = model_id
        self.foundation_model_pipeline = self._current_model.create_pipeline()
        
        logger.info(f"Successfully loaded model: {model_id.value}")

    def _process_prompt_with_guidelines_if_applicable(self, user_prompt: str, target_model_id: ModelId):
        guidelines_config = (
            self._use_zero_shot_chain_of_thought,
            self._use_rag_context
        )
        guidelines_handler = self.guidelines_strategy_map.get(
            guidelines_config,
            # fall back to unfiltered LLM invocation
            self._handle_without_guidelines
        )
        return guidelines_handler(user_prompt, target_model_id)

    def _process_completion_result(self, completion_result: TextGenerationCompletionResult) -> TextGenerationCompletionResult:
        """
        Process guidelines result and create completion result with semantic similarity check.
        
        Args:
            completion_result: Result from text generation
            
        Returns:
            TextGenerationCompletionResult with appropriate completion text
        """

        # analyze the current version of the completion text against prompt injection completions;
        # if guidelines applied, this is the result of completion using guidelines;
        # otherwise it is the raw completion text without guidelines
        completion_result.finalize_completion_text()
        similarity_result: SemanticSimilarityResult = self.semantic_similarity_service.analyze(
            text = completion_result.final_completion_text
        )
        
        # the completion is a result of no guidelines applied 
        if not completion_result.guidelines_result:
            # just return the original
            completion_result.original_result.append_semantic_similarity_result(semantic_similarity_result=similarity_result)
            return completion_result
        
        # completion came from guidelines-enabled service:
        # update completion result with similarity scoring threshold and result
        completion_result.guidelines_result.cosine_similarity_risk_threshold = self.COSINE_SIMILARITY_RISK_THRESHOLD
        completion_result.guidelines_result.append_semantic_similarity_result(semantic_similarity_result=similarity_result)

        # return raw result if the completion comparison score didn't exceed threshold
        if not completion_result.guidelines_result.is_completion_malicious():
            print(f'Guidelines-based completion was NOT malicious. Score: {completion_result.guidelines_result.semantic_similarity_result.max}')
            return completion_result
    
        print(f'Guidelines-based completion was malicious. Score: {completion_result.guidelines_result.semantic_similarity_result.max}')
        completion_result.finalize_completion_text()
        return completion_result

    def _get_template_for_mode(self, mode: GuidelinesMode, target_model_id: Optional[ModelId] = None) -> str:
        """Get the appropriate template ID based on the guidelines mode and model"""
        if target_model_id:
            model_identifier = self._get_model_identifier_from_model_id(target_model_id)
        else:
            model_identifier = self._get_current_model_identifier()
        
        template_map = {
            GuidelinesMode.RAG_PLUS_COT: PromptTemplateType.RAG_PLUS_COT.value,
            GuidelinesMode.COT_ONLY: PromptTemplateType.ZERO_SHOT_COT.value,
            GuidelinesMode.RAG_ONLY: PromptTemplateType.FEW_SHOT.value,
            GuidelinesMode.NONE: PromptTemplateType.BASIC.value
        }
        
        return self._prompt_template_map()[model_identifier][template_map[mode]]

    def _ensure_model_loaded(self, target_model_id: ModelId) -> None:
        """Ensure the correct model is loaded"""
        if (self._current_model_id != target_model_id or 
            self._current_model is None):
            self.load_model(target_model_id)

    def _get_prompt_template(self, template_id: str) -> StringPromptTemplate:
        """Get and validate prompt template"""
        prompt_template = self.prompt_template_service.get(id=template_id)
        if prompt_template is None:
            raise ValueError(f"Prompt template not found for ID: {template_id}")
        return prompt_template

    def _create_guidelines_service(self, mode: GuidelinesMode, prompt_template: StringPromptTemplate) -> AbstractSecurityGuidelinesService:
        """Factory method to create the appropriate guidelines service"""
        base_params = {
            'foundation_model': self._current_model,
            'response_processing_service': self.response_processing_service,
            'prompt_template_service': self.prompt_template_service,
            'llm_configuration_introspection_service': self.llm_configuration_introspection_service
        }
        
        if mode == GuidelinesMode.RAG_PLUS_COT:
            return self.guidelines_factory.create_rag_plus_cot_context_guidelines_service(
                **base_params,
                config_builder=self.guidelines_config_builder
            )
        elif mode == GuidelinesMode.COT_ONLY:
            return self.guidelines_factory.create_cot_guidelines_service(
                **base_params,
                config_builder=self.guidelines_config_builder
            )
        elif mode == GuidelinesMode.RAG_ONLY:
            return self.guidelines_factory.create_rag_context_guidelines_service(**base_params)
        else:
            raise ValueError(f"Unsupported guidelines mode: {mode}")

    def _handle_with_guidelines(self, user_prompt: str, target_model_id: ModelId, mode: GuidelinesMode) -> TextGenerationCompletionResult:
        """Generic handler for guidelines-based processing"""
        # Get template ID and load template
        template_id = self._get_template_for_mode(mode, target_model_id)
        prompt_template = self._get_prompt_template(template_id)
        
        # Ensure correct model is loaded
        self._ensure_model_loaded(target_model_id)
        
        # Create appropriate guidelines service
        guidelines_service = self._create_guidelines_service(mode, prompt_template)
        
        # Apply guidelines and process result
        guidelines_result = guidelines_service.apply_guidelines(user_prompt, template_id)
        return self._process_completion_result(guidelines_result)

    # Simplified handler methods
    def _handle_cot_and_rag(self, user_prompt: str, target_model_id: ModelId) -> TextGenerationCompletionResult:
        """Handle: CoT=True, RAG=True"""
        return self._handle_with_guidelines(user_prompt, target_model_id, GuidelinesMode.RAG_PLUS_COT)

    def _handle_cot_only(self, user_prompt: str, target_model_id: ModelId) -> TextGenerationCompletionResult:
        """Handle: CoT=True, RAG=False"""
        return self._handle_with_guidelines(user_prompt, target_model_id, GuidelinesMode.COT_ONLY)
    
    def _handle_rag_only(self, user_prompt: str, target_model_id: ModelId) -> TextGenerationCompletionResult:
        """Handle: CoT=False, RAG=True"""
        return self._handle_with_guidelines(user_prompt, target_model_id, GuidelinesMode.RAG_ONLY)
    
    def _handle_without_guidelines(self, user_prompt: str) -> TextGenerationCompletionResult:
        """Handle: CoT=False, RAG=False - now with dynamic template selection"""
        try:
            # Get template ID and load template
            template_id = self._get_template_for_mode(GuidelinesMode.NONE)
            prompt_template = self._get_prompt_template(template_id)
            
            print(f'using template: {template_id}')
            
            # Create chain and get config
            chain = self._create_chain_without_guidelines(prompt_template)
            llm_config = self.llm_configuration_introspection_service.get_config(chain)

            # Format prompt
            prompt_value: PromptValue = prompt_template.format_prompt(input=user_prompt)
            prompt_dict = {
                "messages": [
                    {"role": msg.type, "content": msg.content, "additional_kwargs": msg.additional_kwargs}
                    for msg in prompt_value.to_messages()
                ],
                "string_representation": prompt_value.to_string(),
            }

            # Create and return result
            result = TextGenerationCompletionResult(
                original_result=OriginalCompletionResult(
                    user_prompt=user_prompt,
                    completion_text=chain.invoke({self.constants.INPUT_VARIABLE_TOKEN: user_prompt}),
                    llm_config=llm_config,
                    full_prompt=prompt_dict
                )
            )
            return self._process_completion_result(result)
            
        except Exception as e:
            logger.error(f"Error in _handle_without_guidelines: {str(e)}")
            raise e

    # Configuration methods
    def set_config(self, use_cot=False, use_rag=False):
        """Set guidelines configuration"""
        self._use_zero_shot_chain_of_thought = use_cot
        self._use_rag_context = use_rag
        return self
    
    def get_current_config(self):
        """Get current configuration as readable string"""
        return f"CoT: {self._use_zero_shot_chain_of_thought}, RAG: {self._use_rag_context}"

    def without_guidelines(self) -> AbstractTextGenerationCompletionService:
        self._use_guidelines = False
        self._use_zero_shot_chain_of_thought = False
        self._use_rag_context = False
        return self
    
    def with_chain_of_thought_guidelines(self) -> AbstractTextGenerationCompletionService:
        self._use_zero_shot_chain_of_thought = True
        return self
    
    def with_rag_context_guidelines(self) -> AbstractTextGenerationCompletionService:
        self._use_rag_context = True
        return self

    def _create_chain_without_guidelines(self, prompt_template):
        return (
            { f"{self.constants.INPUT_VARIABLE_TOKEN}": RunnablePassthrough() }
            | prompt_template
            | self.foundation_model_pipeline
            | StrOutputParser()
            | self.response_processing_service.process_text_generation_output
        )

    def is_chain_of_thought_enabled(self) -> bool:
        return self._use_zero_shot_chain_of_thought

    def is_rag_context_enabled(self) -> bool:
        return self._use_rag_context

    def add_model_template_mapping(self, model_identifier: str, basic_template_id: str) -> None:
        """
        Add or update a model-to-basic-template mapping.
        
        Args:
            model_identifier: The model identifier/name
            basic_template_id: The corresponding basic template ID
        """
        self._prompt_template_map()[model_identifier.lower()] = basic_template_id

    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model identifiers for basic templates.
        
        Returns:
            list[str]: List of supported model identifiers
        """
        return list(self._prompt_template_map().keys())

    def invoke(self, user_prompt: str, model_id: Optional[ModelId] = None) -> TextGenerationCompletionResult:
        """Generate text using specified or current model"""
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
            
        target_model_id = model_id or self._current_model_id or self.default_model_id
        if (self._current_model_id != target_model_id or 
            self._current_model is None
        ):
            self.load_model(target_model_id)

        print(f'Using model: {target_model_id.value}, guidelines: {self.get_current_config()}')
        completion_result = self._process_prompt_with_guidelines_if_applicable(user_prompt=user_prompt, target_model_id=target_model_id)        
        return completion_result