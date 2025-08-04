import abc


class AbstractFoundationModel(abc.ABC):
    @abc.abstractmethod
    def create_pipeline(self) -> any:
        raise NotImplementedError