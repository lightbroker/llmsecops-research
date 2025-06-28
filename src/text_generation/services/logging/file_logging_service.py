import logging


from src.text_generation.services.logging.abstract_logging_service import AbstractLoggingService


class FileLoggingService(AbstractLoggingService):
    def __init__(self, filename):
        super().__init__(handler=logging.FileHandler(filename))
        self.logger = super()._get_logger()