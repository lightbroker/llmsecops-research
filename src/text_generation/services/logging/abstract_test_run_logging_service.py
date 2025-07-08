import abc
from typing import Any, Dict, List


class AbstractTestRunLoggingService(abc.ABC):
    @abc.abstractmethod
    def log_results(
            self, 
            scores: List[float], 
            mean: float,
            max: float):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_logs(self) -> List[Dict[str, Any]]:
        raise NotImplementedError