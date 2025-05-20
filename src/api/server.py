import json
import logging

from src.api.controller import ApiController
from wsgiref.simple_server import make_server


class RestApiServer:
    def __init__(self):
        pass

    def post_response(self, env, start_response):
        start_response('200 OK', [('Content-Type', 'application/json')])
        yield [json.dumps({'received': 'data'}).encode('utf-8')]

    def listen(self):
        try:
            port = 9999
            controller = ApiController()
            with make_server('', port, controller) as wsgi_srv:
                print(f'listening on port {port}...')
                wsgi_srv.serve_forever()
        except Exception as e:
            logging.warning(e)


if __name__ == '__main__':
    srv = RestApiServer()
    srv.listen()