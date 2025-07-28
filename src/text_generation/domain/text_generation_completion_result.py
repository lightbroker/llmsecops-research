from typing import Optional

from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult
from src.text_generation.domain.alternate_completion_result import AlternateCompletionResult
from src.text_generation.domain.guardrails_result import GuardrailsResult
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.original_completion_result import OriginalCompletionResult


class TextGenerationCompletionResult(AbstractTextGenerationCompletionResult):
    """
    Container class that holds the original completion result and optional 
    guidelines and guardrails processing results.
    """
    
    def __init__(
        self,
        original_result: OriginalCompletionResult,
        guidelines_result: Optional[GuidelinesResult] = None,
        guardrails_result: Optional[GuardrailsResult] = None,
        alternate_result: Optional[AlternateCompletionResult] = None
    ):
        self.original_result = original_result
        self.guidelines_result = guidelines_result
        self.guardrails_result = guardrails_result
        self.alternate_result = alternate_result
        self.final_completion_text = ''

    def finalize_completion_text(self) -> str:
        """
        Returns the current completion text based on priority order:
        1. guardrails_result.completion_text (if not empty)
        2. guidelines_result.completion_text (if not empty)  
        3. original_result.completion_text (if not empty)
        """

        print(f'Finalized text was \"{self.final_completion_text}\"')

        # Check alternate text first
        if (self.alternate_result and 
            self.alternate_result.alterate_completion_text and
            self.alternate_result.alterate_completion_text.strip()
        ):
            self.final_completion_text = self.alternate_result.alterate_completion_text       
            print(f'Using alternate result. Finalized text is now \"{self.final_completion_text}\"')
            return

        # Check guardrails_result.completion_text next
        if (self.guardrails_result and 
            self.guardrails_result.guardrails_completion_text and
            self.guardrails_result.guardrails_completion_text.strip()
        ):
            self.final_completion_text = self.guardrails_result.guardrails_completion_text
            print(f'Using guardrails result. Finalized text is now \"{self.final_completion_text}\"')
            return
        
        # Fall back to guidelines_result.completion_text
        if (self.guidelines_result and 
            self.guidelines_result.guidelines_completion_text and
            self.guidelines_result.guidelines_completion_text.strip()
        ):
            self.final_completion_text = self.guidelines_result.guidelines_completion_text
            print(f'Using guidelines result. Finalized text is now \"{self.final_completion_text}\"')
            return

        # Fall back to original_result.completion_text
        if (self.original_result and 
            self.original_result.completion_text and
            self.original_result.completion_text.strip()
        ):
            self.final_completion_text = self.original_result.completion_text
            print(f'Using original. Finalized text is now \"{self.final_completion_text}\"')
            return
        
        # If all are empty, return empty string
        self.final_completion_text = ""
        print(f'Finalized text is now \"{self.final_completion_text}\"')
        return