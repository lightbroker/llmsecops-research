import abc


class AbstractReflexionSecurityGuidelinesService(abc.ABC):
    """Abstract service for reflexion security guidelines."""
    
    @abc.abstractmethod
    def apply_guidelines(self, context: dict) -> dict:
        """Apply reflexion security guidelines to context."""
        pass
