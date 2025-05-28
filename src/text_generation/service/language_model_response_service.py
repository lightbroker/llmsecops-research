import abc

class AbstractLanguageModelResponseService(abc.ABC):
    @abc.abstractmethod
    def invoke(self, user_input: str) -> str:
        raise NotImplementedError

class LanguageModelResponseService(AbstractLanguageModelResponseService):
    def __call__(self, *args, **kwds):
        pass