from PathDispatcher import PathDispatcher
from wsgiref.simple_server import make_server


class RestApiServer:
    def __init__(self):
        pass

    def response_function(self, environ, start_response):
        start_response('200 OK', [('Content-Type','text/html')])
        yield str(f'testing...\n').encode('utf-8')

    def listen(self):
        port = 9999
        dispatcher = PathDispatcher()
        dispatcher.register('GET', '/hello', self.response_function)
        wsgi_srv = make_server('', port, dispatcher)
        print(f'listening on port {port}...')
        wsgi_srv.serve_forever()


if __name__ == '__main__':
    srv = RestApiServer()
    srv.listen()
