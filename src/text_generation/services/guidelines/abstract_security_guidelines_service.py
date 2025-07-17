import abc


class AbstractSecurityGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def apply_guidelines(self, user_prompt: str) -> str:
        pass


class AbstractRetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder(abc.ABC):
    @abc.abstractmethod
    def get_prompt_template(self) -> str:
        raise NotImplementedError
