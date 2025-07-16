import abc


class AbstractChainOfThoughtSecurityGuidelinesService(abc.ABC):
    """Abstract service for chain of thought security guidelines."""

    @abc.abstractmethod
    def apply_guidelines(self, user_prompt: str) -> str:
        """Apply chain of thought security guidelines to context."""
        pass