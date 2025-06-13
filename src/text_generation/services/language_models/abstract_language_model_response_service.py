import abc


class AbstractLanguageModelResponseService(abc.ABC):
    @abc.abstractmethod
    def invoke(self, user_prompt: str) -> str:
        raise NotImplementedError