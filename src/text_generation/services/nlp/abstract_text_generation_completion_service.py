import abc

from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class AbstractTextGenerationCompletionService(abc.ABC):

    @abc.abstractmethod
    def without_guidelines(self) -> 'AbstractTextGenerationCompletionService':
        """Skip all security guidelines."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def with_chain_of_thought_guidelines(self) -> 'AbstractTextGenerationCompletionService':
        """Enable chain of thought security guidelines."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def with_rag_context_guidelines(self) -> 'AbstractTextGenerationCompletionService':
        """Enable RAG context security guidelines."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def with_reflexion_guidelines(self) -> 'AbstractTextGenerationCompletionService':
        """Enable reflexion security guidelines."""
        raise NotImplementedError

    @abc.abstractmethod
    def invoke(self, user_prompt: str) -> AbstractTextGenerationCompletionResult:
        raise NotImplementedError