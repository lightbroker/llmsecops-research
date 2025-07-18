import abc

from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class AbstractSecurityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def apply_guidelines(self, user_prompt: str) -> AbstractGuidelinesProcessedCompletion:
        pass


class AbstractRetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder(abc.ABC):
    @abc.abstractmethod
    def get_prompt_template(self) -> str:
        raise NotImplementedError
