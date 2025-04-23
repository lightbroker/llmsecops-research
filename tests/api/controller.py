import json
import os
import sys

# Add the parent folder (or any relative path)
sys.path.append(os.path.abspath('./../llm'))

from phi3_language_model import Phi3LanguageModel


class ApiController:
    def __init__(self):
        self.routes = {}


    def __http_415_notsupported(self, env, start_response):
        start_response('415 Unsupported Media Type', self.response_headers)
        return [json.dumps({'error': 'Unsupported Content-Type'}).encode('utf-8')]

    def get_service_response(self, prompt):
        service = Phi3LanguageModel()
        response = service.get_response(prompt_input=prompt)
        return response

    def __http_200_ok(self, env, start_response):
        try:
            request_body_size = int(env.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = env['wsgi.input'].read(request_body_size)
        request_json = json.loads(request_body.decode('utf-8'))
        prompt = request_json.get('prompt')
        
        # for now, just reading request and echoing back in response
        # data = json.loads(prompt)
        # response_body = json.dumps(data).encode('utf-8')

        data = self.get_service_response(prompt)
        response_body = json.dumps(data).encode('utf-8')
        
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]


    def __call__(self, env, start_response):
        method = env.get('REQUEST_METHOD').upper()
        path = env.get('PATH_INFO')

        # TODO: register route for POST /api/conversations

        if not method == 'POST':
            self.__http_415_notsupported(env, start_response)

        try:                
            handler = self.routes.get((method,path), self.__http_200_ok)
            return handler(env, start_response)
        except json.JSONDecodeError:
            start_response('400 Bad Request', self.response_headers)
            return [json.dumps({'error': 'Invalid JSON'}).encode('utf-8')]

