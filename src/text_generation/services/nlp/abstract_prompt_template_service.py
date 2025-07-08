import abc


class AbstractPromptTemplateService(abc.ABC):
    @abc.abstractmethod
    def get(self, id: str) -> abc.ABC:
        raise NotImplementedError