from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.text_generation.common.constants import Constants
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel


class TextGenerationCompletionService(
        AbstractTextGenerationCompletionService):
    def __init__(
            self, 
            foundation_model: AbstractFoundationModel,
            prompt_template_service: AbstractPromptTemplateService,
            chain_of_thought_service: AbstractChainOfThoughtSecurityGuidelinesService,
            rag_context_service: AbstractRetrievalAugmentedGenerationContextSecurityGuidelinesService,
            reflexion_service: AbstractReflexionSecurityGuidelinesService):
        super().__init__()
        self.constants = Constants()
        
        self._language_model_pipeline = foundation_model.create_pipeline()
        self._prompt_template_service = prompt_template_service
        self._chain_of_thought_service = chain_of_thought_service
        self._rag_context_service = rag_context_service
        self._reflexion_service = reflexion_service

        self._use_guidelines = True
        self._use_chain_of_thought = True
        self._use_rag_context = True
        self._use_reflexion = True

    def _extract_assistant_response(self, text):
        if self.constants.ASSISTANT_TOKEN in text:
            return text.split(self.constants.ASSISTANT_TOKEN)[-1].strip()
        return text

    def without_guidelines(self) -> AbstractTextGenerationCompletionService:
        """Skip all security guidelines."""
        self._use_guidelines = False
        return self
    
    def with_chain_of_thought_guidelines(self) -> AbstractTextGenerationCompletionService:
        """Enable chain-of-thought (CoT) security guidelines."""
        self._use_chain_of_thought = True
        return self
    
    def with_rag_context_guidelines(self) -> AbstractTextGenerationCompletionService:
        """Enable RAG-enriched examples context security guidelines."""
        self._use_rag_context = True
        return self
    
    def with_reflexion_guidelines(self) -> AbstractTextGenerationCompletionService:
        """Enable reflexion security guidelines."""
        self._use_reflexion = True
        return self

    def create_chain(self):
        prompt_template_id=self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_BASIC
        prompt_template = self._prompt_template_service.get(id=prompt_template_id)
        return (
            { "question": RunnablePassthrough() }
            | prompt_template
            | self._language_model_pipeline
            | StrOutputParser()
            | self._extract_assistant_response
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

        # security guidelines combinations
        if self._use_chain_of_thought and self._use_rag_context and self._use_reflexion:
            # All three enabled: CoT + RAG + Reflexion
            pass
        elif self._use_chain_of_thought and self._use_rag_context and not self._use_reflexion:
            # CoT + RAG only
            pass
        elif self._use_chain_of_thought and not self._use_rag_context and self._use_reflexion:
            # CoT + Reflexion only
            pass
        elif self._use_chain_of_thought and not self._use_rag_context and not self._use_reflexion:
            # CoT only
            pass
        elif not self._use_chain_of_thought and self._use_rag_context and self._use_reflexion:
            # RAG + Reflexion only
            pass
        elif not self._use_chain_of_thought and self._use_rag_context and not self._use_reflexion:
            # RAG only
            pass
        elif not self._use_chain_of_thought and not self._use_rag_context and self._use_reflexion:
            # Reflexion only
            pass
        else:
            # None enabled (all False)
            pass