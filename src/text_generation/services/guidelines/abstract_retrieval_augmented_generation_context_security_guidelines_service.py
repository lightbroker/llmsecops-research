import abc


class AbstractRetrievalAugmentedGenerationContextSecurityGuidelinesService(abc.ABC):
    """Abstract service for RAG context security guidelines."""

    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply RAG context security guidelines to context."""
        pass