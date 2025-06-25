import abc


class AbstractEmbeddingModel(abc.ABC):
    @property
    @abc.abstractmethod
    def embeddings(self):
        raise NotImplementedError