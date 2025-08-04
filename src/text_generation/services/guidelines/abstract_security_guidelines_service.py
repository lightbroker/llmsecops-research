import abc

from langchain_core.prompts import StringPromptTemplate

from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion


class AbstractSecurityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def apply_guidelines(self, user_prompt: str) -> AbstractGuidelinesProcessedCompletion:
        pass


class AbstractSecurityGuidelinesConfigurationBuilder(abc.ABC):
    @abc.abstractmethod
    def get_prompt_template(self, template_id: str, user_prompt: str) -> StringPromptTemplate:
        raise NotImplementedError
