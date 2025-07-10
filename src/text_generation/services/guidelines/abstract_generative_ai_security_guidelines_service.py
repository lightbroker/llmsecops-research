import abc

from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class AbstractGenerativeAiSecurityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def for_prompt(self, prompt: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def skip_guidelines(self):
        raise NotImplementedError

    @abc.abstractmethod
    def use_chain_of_thought(self):
        raise NotImplementedError

    @abc.abstractmethod
    def use_examples_from_rag(self):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self) -> AbstractGuidelinesProcessedCompletion:
        raise NotImplementedError