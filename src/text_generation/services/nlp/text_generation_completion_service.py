from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_huggingface import HuggingFacePipeline

from src.text_generation.common.constants import Constants
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


class TextGenerationCompletionService(
        AbstractTextGenerationCompletionService):
    def __init__(
            self, 
            foundation_model: AbstractFoundationModel,
            response_processing_service: AbstractResponseProcessingService,
            prompt_template_service: AbstractPromptTemplateService,
            chain_of_thought_guidelines: AbstractSecurityGuidelinesService,
            rag_context_guidelines: AbstractSecurityGuidelinesService,
            rag_plus_cot_guidelines: AbstractSecurityGuidelinesService,
            reflexion_guardrails: AbstractGeneratedTextGuardrailService,
            semantic_similarity_service: AbstractSemanticSimilarityService,
            prompt_injection_example_service: AbstractPromptInjectionExampleService,
            llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service

        # set up semantic similarity service and supporting texts
        self.example_prompt_injection_completions = prompt_injection_example_service.get_all_completions()
        self.example_prompt_injection_prompts = prompt_injection_example_service.get_all_prompts()
        self.semantic_similarity_service = semantic_similarity_service
        self.semantic_similarity_service.use_comparison_texts(
            self.example_prompt_injection_completions
        )

        # guidelines services
        self.chain_of_thought_guidelines: AbstractSecurityGuidelinesService = chain_of_thought_guidelines
        self.rag_context_guidelines: AbstractSecurityGuidelinesService = rag_context_guidelines
        self.rag_plus_cot_guidelines: AbstractSecurityGuidelinesService = rag_plus_cot_guidelines
        
        # guardrails services
        self.reflexion_guardrails: AbstractGeneratedTextGuardrailService = reflexion_guardrails

        # constants
        self.COSINE_SIMILARITY_RISK_THRESHOLD = 0.8

        # default guidelines settings
        self._use_guidelines = False
        self._use_zero_shot_chain_of_thought = False
        self._use_rag_context = False

        # dictionary dispatch for handling guidelines combinations
        self.guidelines_strategy_map = {
            (True, True):   self._handle_cot_and_rag,
            (True, False):  self._handle_cot_only,
            (False, True):  self._handle_rag_only,
            (False, False): self._handle_without_guidelines,
        }

        # default guardrails settings
        self._use_reflexion_guardrails = False

        # introspection for logging
        self.llm_configuration_introspection_service = llm_configuration_introspection_service


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
            chain = self._create_chain_without_guidelines()
            llm_config = self.llm_configuration_introspection_service.get_config(chain)

            result = TextGenerationCompletionResult(
                original_result=OriginalCompletionResult(
                    user_prompt=user_prompt,
                    completion_text=chain.invoke(user_prompt),
                    llm_config=llm_config
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

    def _create_chain_without_guidelines(self):
        prompt_template = self.prompt_template_service.get(
            id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__01_BASIC
        )

        if prompt_template is None:
            raise ValueError(f"Prompt template not found for ID: {self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__01_BASIC}")
    
        return (
            { "question": RunnablePassthrough() }
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


    def invoke(self, user_prompt: str) -> TextGenerationCompletionResult:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        print(f'Using guidelines: {self.get_current_config()}')
        completion_result: TextGenerationCompletionResult = self._process_prompt_with_guidelines_if_applicable(user_prompt)
        
        if not self._use_reflexion_guardrails:
            return completion_result
        
        return self._handle_reflexion_guardrails(completion_result)
