import abc
from typing import Dict, List


class AbstractPromptInjectionExampleRepository(abc.ABC):
    @abc.abstractmethod
    def get_all(self) -> List[Dict[str, str]]:
        raise NotImplementedError