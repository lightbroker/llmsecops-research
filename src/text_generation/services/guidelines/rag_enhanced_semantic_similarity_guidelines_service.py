import abc


class AbstractChainOfThoughtSecurityGuidelinesService(abc.ABC):
    """Abstract service for chain of thought security guidelines."""
    
    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply chain of thought security guidelines to context."""
        pass


class AbstractRetrievalAugmentedGenerationContextSecurityGuidelinesService(abc.ABC):
    """Abstract service for RAG context security guidelines."""
    
    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply RAG context security guidelines to context."""
        pass


class AbstractReflexionSecurityGuidelinesService(abc.ABC):
    """Abstract service for reflexion security guidelines."""
    
    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply reflexion security guidelines to context."""
        pass