import logging

class LoggingService:
    # TODO use base class and make this file logging service

    def __init__(self, filename):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.FileHandler(filename))
        self.logger = logger
