from enum import Enum
from typing import Optional, Dict, Any
import logging

from langchain.prompts import StringPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompt_values import PromptValue

from src.text_generation.adapters.foundation_models.base.base_model_config import BaseModelConfig
from src.text_generation.adapters.foundation_models.factories.foundation_model_factory import FoundationModelFactory
from src.text_generation.common.constants import Constants
from src.text_generation.common.model_id import ModelId
from src.text_generation.domain.alternate_completion_result import AlternateCompletionResult
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.original_completion_result import OriginalCompletionResult
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.prompt_injection.abstract_prompt_injection_example_service import AbstractPromptInjectionExampleService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


logger = logging.getLogger(__name__)

class TextGenerationCompletionService(
        AbstractTextGenerationCompletionService):
    def __init__(
            self, 
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            chain_of_thought_guidelines: AbstractSecurityGuidelinesService,
            rag_context_guidelines: AbstractSecurityGuidelinesService,
            rag_plus_cot_guidelines: AbstractSecurityGuidelinesService,
            reflexion_guardrails: AbstractGeneratedTextGuardrailService,
            semantic_similarity_service: AbstractSemanticSimilarityService,
            prompt_injection_example_service: AbstractPromptInjectionExampleService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
            default_model_type: ModelId = ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT):
        
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
        self.chain_of_thought_guidelines = chain_of_thought_guidelines
        self.rag_context_guidelines = rag_context_guidelines
        self.rag_plus_cot_guidelines = rag_plus_cot_guidelines
        
        # Guardrails service
        self.reflexion_guardrails = reflexion_guardrails

        # Constants and settings
        self.COSINE_SIMILARITY_RISK_THRESHOLD = 0.8
        self._use_guidelines = False
        self._use_zero_shot_chain_of_thought = False
        self._use_rag_context = False
        self._use_reflexion_guardrails = False

        # Strategy map for guidelines
        self.guidelines_strategy_map = {
            (True, True):   self._handle_cot_and_rag,
            (True, False):  self._handle_cot_only,
            (False, True):  self._handle_rag_only,
            (False, False): self._handle_without_guidelines,
        }

        # Load default model
        self.load_model(default_model_type)


    def load_model(
        self, 
        model_id: ModelId, 
        config: Optional[BaseModelConfig] = None,
        force_reload: bool = False
    ) -> None:
        """Load a specific model"""
        if (not force_reload and 
            self._current_model is not None and 
            self._current_model_id == model_id and
            self._current_model.is_loaded()):
            logger.info(f"Model {model_id.value} already loaded")
            return
        
        if self._current_model is not None:
            self._current_model.unload()
        
        self._current_model = self.factory.create_model(model_id, config)
        self._current_model.load()
        self._current_model_id: ModelId = model_id
        self.foundation_model_pipeline = self._current_model.create_pipeline()
        
        logger.info(f"Successfully loaded model: {model_id.value}")

    def switch_model(self, model_id: ModelId, config: Optional[BaseModelConfig] = None) -> None:
        """Switch to a different model"""
        self.load_model(model_id, config, force_reload=True)

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded model"""
        if self._current_model and self._current_model.is_loaded():
            return self._current_model.get_model_info()
        return None


    def _process_prompt_with_guidelines_if_applicable(self, user_prompt: str):
        guidelines_config = (
            self._use_zero_shot_chain_of_thought,
            self._use_rag_context
        )
        guidelines_handler = self.guidelines_strategy_map.get(
            guidelines_config,

            # fall back to unfiltered LLM invocation
            self._handle_without_guidelines
        )
        return guidelines_handler(user_prompt)


    def _process_completion_result(self, completion_result: TextGenerationCompletionResult) -> TextGenerationCompletionResult:
        """
        Process guidelines result and create completion result with semantic similarity check.
        
        Args:
            guidelines_result: Result from applying security guidelines
            
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


    # Handler methods for each guidelines combination
    def _handle_cot_and_rag(self, user_prompt: str) -> TextGenerationCompletionResult:
        """Handle: CoT=True, RAG=True"""
        guidelines_result = self.rag_plus_cot_guidelines.apply_guidelines(user_prompt)
        return self._process_completion_result(guidelines_result)

    def _handle_cot_only(self, user_prompt: str) -> TextGenerationCompletionResult:
        """Handle: CoT=True, RAG=False"""
        guidelines_result = self.chain_of_thought_guidelines.apply_guidelines(user_prompt)
        return self._process_completion_result(guidelines_result)
    
    def _handle_rag_only(self, user_prompt: str) -> TextGenerationCompletionResult:
        """Handle: CoT=False, RAG=True"""
        guidelines_result = self.rag_context_guidelines.apply_guidelines(user_prompt)
        return self._process_completion_result(guidelines_result)
    
    def _handle_without_guidelines(self, user_prompt: str) -> TextGenerationCompletionResult:
        """Handle: CoT=False, RAG=False"""
        try:
            prompt_template: StringPromptTemplate = self.prompt_template_service.get(
                id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__01_BASIC
            )

            if prompt_template is None:
                raise ValueError(f"Prompt template not found for ID: {self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__01_BASIC}")
            
            chain = self._create_chain_without_guidelines(prompt_template)
            llm_config = self.llm_configuration_introspection_service.get_config(chain)

            prompt_value: PromptValue = prompt_template.format_prompt(input=user_prompt)
            prompt_dict = {
                "messages": [
                    {"role": msg.type, "content": msg.content, "additional_kwargs": msg.additional_kwargs}
                    for msg in prompt_value.to_messages()
                ],
                "string_representation": prompt_value.to_string(),
            }

            result = TextGenerationCompletionResult(
                original_result=OriginalCompletionResult(
                    user_prompt=user_prompt,
                    completion_text=chain.invoke({ self.constants.INPUT_VARIABLE_TOKEN: user_prompt }),
                    llm_config=llm_config,
                    full_prompt=prompt_dict
            ))
            return self._process_completion_result(result)
        except Exception as e:
            raise e

    def _handle_reflexion_guardrails(self, text_generation_completion_result: TextGenerationCompletionResult) -> TextGenerationCompletionResult:
        result_with_guardrails_applied = self.reflexion_guardrails.apply_guardrails(text_generation_completion_result)
        return result_with_guardrails_applied

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

    def with_reflexion_guardrails(self) -> AbstractTextGenerationCompletionService:
        self._use_reflexion_guardrails = True
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

    def is_reflexion_enabled(self) -> bool:
        return self._use_reflexion_guardrails


    def invoke(self, user_prompt: str, model_id: Optional[ModelId] = None) -> TextGenerationCompletionResult:
        """Generate text using specified or current model"""
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
            
        target_model_id = model_id or self._current_model_id or self.default_model_id
        if (self._current_model_id != target_model_id or 
            self._current_model is None or 
            not self._current_model.is_loaded()):
            self.load_model(target_model_id)

        print(f'Using model: {target_model_id.value}, guidelines: {self.get_current_config()}')
        completion_result = self._process_prompt_with_guidelines_if_applicable(user_prompt)
        
        if not self._use_reflexion_guardrails:
            return completion_result
        
        return self._handle_reflexion_guardrails(completion_result)
