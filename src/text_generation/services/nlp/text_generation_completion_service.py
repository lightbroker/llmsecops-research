from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.text_generation.common.constants import Constants
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesService
from src.text_generation.services.guidelines.chain_of_thought_security_guidelines_service import ChainOfThoughtSecurityGuidelinesService
from src.text_generation.services.guardrails.reflexion_security_guidelines_service import ReflexionSecurityGuardrailsService
from src.text_generation.services.guidelines.rag_context_security_guidelines_service import RetrievalAugmentedGenerationContextSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
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
            reflexion_guardrails: AbstractGeneratedTextGuardrailService):
        super().__init__()
        self.constants = Constants()
        self.foundation_model_pipeline = foundation_model.create_pipeline()
        self.response_processing_service = response_processing_service
        self.prompt_template_service = prompt_template_service

        # guidelines services
        self.chain_of_thought_guidelines: AbstractSecurityGuidelinesService = chain_of_thought_guidelines
        self.rag_context_guidelines: AbstractSecurityGuidelinesService = rag_context_guidelines
        
        # guardrails services
        self.reflexion_guardrails: AbstractSecurityGuidelinesService = reflexion_guardrails

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
        self._use_reflexion = False

    def _process_prompt_with_guidelines_if_applicable(self, user_prompt: str):
        guidelines_config = (
            self._use_zero_shot_chain_of_thought,
            self._use_rag_context
        )
        guidelines_handler = self.guidelines_strategy_map.get(
            guidelines_config, 
            self._handle_without_guidelines
        )
        return guidelines_handler(user_prompt)

    # Handler methods for each combination
    def _handle_cot_and_rag(self, user_prompt: str):
        """Handle: CoT=True, RAG=True"""
        context = self._retrieve_rag_context(query)
        thought_process = self._apply_chain_of_thought(query, context)
        return f"CoT+RAG: {thought_process}"
    
    def _handle_cot_only(self, user_prompt: str):
        """Handle: CoT=True, RAG=False"""
        print("🧠 Using Chain of Thought only")
        thought_process = self._apply_chain_of_thought(query)
        return f"CoT: {thought_process}"
    
    def _handle_rag_only(self, user_prompt: str):
        """Handle: CoT=False, RAG=True"""
        self.rag_context_guidelines.apply_guidelines(user_prompt)
        return f"RAG: {response}"
    
    def _handle_without_guidelines(self, user_prompt: str):
        """Handle: CoT=False, RAG=False"""
        response = self._basic_generate(query)
        return f"Basic: {response}"
    
    # Helper methods (example implementations)
    def _retrieve_rag_context(self, user_prompt: str):
        """Retrieve relevant context from knowledge base"""
        return f"Context for '{query}'"
    
    def _apply_chain_of_thought(self, user_prompt: str):
        """Apply chain of thought reasoning"""
        if context:
            return f"Step-by-step reasoning for '{query}' with {context}"
        return f"Step-by-step reasoning for '{query}'"
    
    def _generate_with_context(self, user_prompt: str):
        """Generate response using context"""
        return f"Response to '{query}' using {context}"
    
    def _basic_generate(self, user_prompt: str):
        """Basic generation without special features"""
        return f"Basic response to '{query}'"
    
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
        self._use_reflexion = True
        return self

    def create_chain(self):
        prompt_template = self.prompt_template_service.get(
            id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_BASIC
        )
        return (
            { "question": RunnablePassthrough() }
            | prompt_template
            | self.foundation_model_pipeline
            | StrOutputParser()
            | self.response_processing_service.process_text_generation_output
        )

    def invoke(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")
        
        if self._use_guidelines == False:
            try:
                chain = self.create_chain()
                return chain.invoke(user_prompt)
            except Exception as e:
                raise e

        self._process_prompt_with_guidelines_if_applicable(user_prompt)