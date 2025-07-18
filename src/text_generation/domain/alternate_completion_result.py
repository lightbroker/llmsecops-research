from src.text_generation.domain.abstract_text_generation_completion_result import AbstractTextGenerationCompletionResult


class AlternateCompletionResult(AbstractTextGenerationCompletionResult):
    """
        Represents an alternative text completion based on 
        security guidelines processing results.
    """
    def __init__(
            self, 
            alterate_completion_text: str):
        self.alterate_completion_text = alterate_completion_text