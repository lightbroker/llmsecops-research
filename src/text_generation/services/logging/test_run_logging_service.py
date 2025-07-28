import calendar
import json
import os
import threading
import time

from datetime import datetime
from typing import Any, Dict, List

from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.services.logging.abstract_test_run_logging_service import AbstractTestRunLoggingService


class TestRunLoggingService(AbstractTestRunLoggingService):
    def __init__(self, test_id: int):
        self._lock = threading.Lock()
        timestamp = calendar.timegm(time.gmtime())
        self.log_file_path = f"./tests/logs/test_{test_id}/test_{test_id}_logs_{timestamp}.json"
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
            text_generation_completion_result: TextGenerationCompletionResult,
            final_completion_text_score: float,
            is_rag_few_shot_enabled: bool,
            is_cot_enabled: bool,
            is_reflexion_enabled: bool,
            original_llm_config: dict):
        with self._lock:
            logs = self._read_logs()
            log_entry = {
                "id": id,
                "timestamp": datetime.now().isoformat(),
                "final_completion_text_score": final_completion_text_score,
                "mitigations_enabled": {
                    "guidelines": {
                        "rag_with_few_shot_examples": is_rag_few_shot_enabled,
                        "chain_of_thought": is_cot_enabled
                    },
                    "guardrails": {
                        "reflexion": is_reflexion_enabled
                    }
                },
                "original_llm_config": original_llm_config,
                "text_generation_completion_result": text_generation_completion_result.__dict__
            }
            logs.append(log_entry)
            self._write_logs(logs)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_logs()