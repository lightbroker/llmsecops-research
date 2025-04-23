import json

from PathDispatcher import PathDispatcher
from wsgiref.simple_server import make_server


class RestApiServer:
    def __init__(self):
        pass

    def post_response(self, env, start_response):
        start_response('200 OK', [('Content-Type', 'application/json')])
        yield [json.dumps({'received': 'data'}).encode('utf-8')]

    def listen(self):
        port = 9999
        dispatcher = PathDispatcher()
        dispatcher.register('POST', '/', self.post_response)
        with make_server('', port, dispatcher) as wsgi_srv:
            print(f'listening on port {port}...')
            wsgi_srv.serve_forever()


if __name__ == '__main__':
    srv = RestApiServer()
    srv.listen()