import abc
from typing import Any, Dict, List


class AbstractWebTrafficLoggingService(abc.ABC):
    @abc.abstractmethod
    def log_request_response(self, request: str, response: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_logs(self) -> List[Dict[str, Any]]:
        raise NotImplementedError