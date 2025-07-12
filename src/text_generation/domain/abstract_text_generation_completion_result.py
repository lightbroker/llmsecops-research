import abc

class AbstractTextGenerationCompletionResult(abc.ABC):
    
    @abc.abstractmethod
    def get_text(self) -> str:
        """Return the generated text."""
        pass
    
    @abc.abstractmethod
    def get_metadata(self) -> dict:
        """Return metadata about the generation."""
        pass
