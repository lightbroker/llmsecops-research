import abc


class AbstractRetrievalAugmentedGenerationGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def get_prompt_template(self) -> str:
        raise NotImplementedError
    
    @abc.abstractmethod
    def create_guidelines_context(self, user_prompt: str) -> str:
        raise NotImplementedError