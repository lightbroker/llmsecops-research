from abc import ABC, abstractmethod
from typing import List

class AbstractPromptInjectionExampleService(ABC):
    """Abstract base class for prompt injection example services."""
    
    @abstractmethod
    def get_all_prompts(self) -> List[str]:
        """Get all prompt injection prompts."""
        pass
        
    @abstractmethod
    def get_all_completions(self) -> List[str]:
        """Get all prompt injection completions."""
        pass