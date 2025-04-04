import cgi


class PathDispatcher:
    def __init__(self):
        self.routes = {}

    def notfound_404(self, env, start_response):
        start_response('404 Not Found', [ ('Content-Type', 'text/plain') ])
        return [b'Not Found']

    def __call__(self, env, start_response):
        path = env.get('PATH_INFO')
        params = cgi.FieldStorage(env.get('wsgi.output'), environ=env)
        method = env.get('REQUEST_METHOD').lower()
        env['params'] = { key: params.getvalue(key) for key in params }
        handler = self.routes.get((method,path), self.notfound_404)
        return handler(env, start_response)
    
    def register(self, method, path, function):
        self.routes[method.lower(), path] = function
        return function
