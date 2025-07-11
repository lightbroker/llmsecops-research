import abc

from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class AbstractGenerativeAiSecurityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def for_prompt(self, prompt: str):
        return self
    
    @abc.abstractmethod
    def skip_guidelines(self):
        return self

    @abc.abstractmethod
    def use_chain_of_thought(self):
        return self

    @abc.abstractmethod
    def use_examples_from_rag(self):
        return self

    @abc.abstractmethod
    def apply(self) -> AbstractGuidelinesProcessedCompletion:
        raise NotImplementedError