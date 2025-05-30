from src.text_generation.entrypoints.http_api_controller import HttpApiController
from src.text_generation.service.logging_service import LoggingService
from wsgiref.simple_server import make_server


class RestApiServer:
    def __init__(self):
        self.logger = LoggingService(filename='text_generation.server.log').logger

    def listen(self):
        try:
            port = 9999
            controller = HttpApiController()
            with make_server('', port, controller) as wsgi_srv:
                print(f'listening on port {port}...')
                wsgi_srv.serve_forever()
        except Exception as e:
            self.logger.debug(e)

if __name__ == '__main__':
    srv = RestApiServer()
    srv.listen()