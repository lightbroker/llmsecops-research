import abc


class AbstractLLMConfigurationIntrospectionService(abc.ABC):
    @abc.abstractmethod
    def get_config(self, chain) -> dict:
        raise NotImplementedError