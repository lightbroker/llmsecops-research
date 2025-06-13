import logging
import pytest

from src.text_generation.services.logging.file_logging_service import FileLoggingService
from src.text_generation.services.language_models.fake_language_model_response_service import FakeLanguageModelResponseService


def test_file_logging_service_has_filehandler():
    logfile = 'test.log'
    svc = FileLoggingService(filename=logfile)
    assert svc.logger.hasHandlers() == True
    assert any(type(handler) == logging.FileHandler for handler in svc.logger.handlers)


def test_language_model_response_service_valid_input():
    svc = FakeLanguageModelResponseService()
    response = svc.invoke('what is 1 + 1?')
    assert response != None
    assert response != ''


def test_language_model_response_service_empty_input():
    svc = FakeLanguageModelResponseService()

    with pytest.raises(ValueError):
        _ = svc.invoke(user_prompt='')

    with pytest.raises(ValueError):
        user_prompt = None
        _ = svc.invoke(user_prompt=user_prompt)
