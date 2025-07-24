import calendar
import json
import os
import threading
import time

from datetime import datetime
from typing import Any, Dict, List

from src.text_generation.services.logging.abstract_test_run_logging_service import AbstractTestRunLoggingService


class TestRunLoggingService(AbstractTestRunLoggingService):
    def __init__(self, test_id: int):
        self._lock = threading.Lock()
        timestamp = calendar.timegm(time.gmtime())
        self.log_file_path = f"./tests/logs/test_{test_id}_logs_{timestamp}.json"
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

    def log_results(
            self, 
            id: str,
            prompt: str,
            completion: str,
            is_rag_few_shot_enabled: bool,
            is_cot_enabled: bool,
            is_reflexion_enabled: bool,
            scores: List[float],
            mean: float,
            max: float):
        with self._lock:
            logs = self._read_logs()
            log_entry = {
                "id": id,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "completion": completion,
                "is_rag_few_shot_enabled": is_rag_few_shot_enabled,
                "is_cot_enabled": is_cot_enabled,
                "is_reflexion_enabled": is_reflexion_enabled,
                "mean": mean,
                "max": max,
                "scores": scores
            }
            logs.append(log_entry)
            self._write_logs(logs)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_logs()