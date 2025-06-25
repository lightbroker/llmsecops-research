import abc


class AbstractResponseProcessingService(abc.ABC):
    @abc.abstractmethod
    def process_text_generation_output(self, output: str) -> str:
        raise NotImplementedError