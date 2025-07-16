import abc


class AbstractPromptInjectionExampleSecurityGuidelinesService(abc.ABC):
    """Abstract service for prompt injection few shot example-based security guidelines."""

    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply RAG context security guidelines to context."""
        pass