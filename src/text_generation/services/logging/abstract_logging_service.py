import abc
import logging


class AbstractLoggingService(abc.ABC):
    def __init__(self, handler: logging.Handler):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.logger = logger

    def _get_logger(self):
        return self.logger