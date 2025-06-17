from wsgiref.simple_server import make_server

from src.text_generation.entrypoints.http_api_controller import HttpApiController
from src.text_generation.services.logging.file_logging_service import FileLoggingService


class RestApiServer:
    def __init__(
            self, 
            listening_port: int, 
            logging_service: FileLoggingService,
            api_controller: HttpApiController
    ):
        self.listening_port = listening_port
        self.logger = logging_service.logger
        self.api_controller = api_controller

    def listen(self):
        try:
            with make_server('', self.listening_port, self.api_controller) as wsgi_srv:
                print(f'listening on port {self.listening_port}...')
                wsgi_srv.serve_forever()
        except Exception as e:
            self.logger.debug(e)