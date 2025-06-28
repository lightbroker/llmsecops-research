import calendar
import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from src.text_generation.services.logging.abstract_web_traffic_logging_service import AbstractWebTrafficLoggingService


class JSONWebTrafficLoggingService(AbstractWebTrafficLoggingService):
    def __init__(self):
        self._lock = threading.Lock()
        timestamp = calendar.timegm(time.gmtime())
        self.log_file_path = f"http_logs_{timestamp}.json"
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w') as f:
                json.dump([], f)

    def _read_logs(self) -> List[Dict[str, Any]]:
        try:
            with open(self.log_file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_logs(self, logs: List[Dict[str, Any]]):
        with open(self.log_file_path, 'w') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def log_request_response(
            self, 
            request: str, 
            response: str):
        with self._lock:
            logs = self._read_logs()
            log_entry = {
                "request": request,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            logs.append(log_entry)
            self._write_logs(logs)

    def get_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_logs()