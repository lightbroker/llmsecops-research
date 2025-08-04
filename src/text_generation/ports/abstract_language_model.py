import abc


class AbstractLanguageModel(abc.ABC):
    @abc.abstractmethod
    def invoke(self, user_prompt: str) -> str:
        raise NotImplementedError