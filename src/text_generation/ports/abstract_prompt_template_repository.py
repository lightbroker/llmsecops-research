import abc


class AbstractPromptTemplateRepository(abc.ABC):
    @abc.abstractmethod
    def get(self, id: str) -> abc.ABC:
        raise NotImplementedError
    
    @abc.abstractmethod
    def add(self, id: str, prompt_template: abc.ABC) -> None:
        raise NotImplementedError