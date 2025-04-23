import cgi
import json


class PathDispatcher:
    def __init__(self):
        self.routes = {}


    def __http_415_notsupported(self, env, start_response):
        start_response('415 Unsupported Media Type', self.response_headers)
        return [json.dumps({'error': 'Unsupported Content-Type'}).encode('utf-8')]


    def __http_200_ok(self, env, start_response):
        try:
            request_body_size = int(env.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = env['wsgi.input'].read(request_body_size)
        request_body = request_body.decode('utf-8')
        
        # for now, just reading request and echoing back in response
        data = json.loads(request_body)
        response_body = json.dumps(data).encode('utf-8')
        
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]


    def __call__(self, env, start_response):
        method = env.get('REQUEST_METHOD').upper()
        path = env.get('PATH_INFO')

        if not method == 'POST':
            self.__http_415_notsupported(env, start_response)

        try:                
            handler = self.routes.get((method,path), self.__http_200_ok)
            return handler(env, start_response)
        except json.JSONDecodeError:
            start_response('400 Bad Request', self.response_headers)
            return [json.dumps({'error': 'Invalid JSON'}).encode('utf-8')]


    def register(self, method, path, function):
        self.routes[method.lower(), path] = function
        return function
