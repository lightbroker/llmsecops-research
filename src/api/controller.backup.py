import json
import traceback

from src.llm.llm import Phi3LanguageModel
from src.llm.llm_rag import Phi3LanguageModelWithRag

class ApiController:
    def __init__(self):
        self.routes = {}
        # Register routes
        self.register_routes()

    def register_routes(self):
        """Register all API routes"""
        self.routes[('POST', '/api/conversations')] = self.handle_conversations
        self.routes[('POST', '/api/rag_conversations')] = self.handle_conversations_with_rag

    def __http_415_notsupported(self, env, start_response):
        response_headers = [('Content-Type', 'application/json')]
        start_response('415 Unsupported Media Type', response_headers)
        return [json.dumps({'error': 'Unsupported Content-Type'}).encode('utf-8')]

    def get_service_response(self, prompt):
        service = Phi3LanguageModel()
        response = service.invoke(user_input=prompt)
        return response
    
    def get_service_response_with_rag(self, prompt):
        service = Phi3LanguageModelWithRag()
        response = service.invoke(user_input=prompt)
        return response

    def format_response(self, data):
        """Format response data as JSON with 'response' key"""
        response_data = {'response': data}
        try:
            response_body = json.dumps(response_data).encode('utf-8')
        except:
            # If serialization fails, convert data to string first
            response_body = json.dumps({'response': str(data)}).encode('utf-8')
        return response_body

    def handle_conversations(self, env, start_response):
        """Handle POST requests to /api/conversations"""
        try:
            request_body_size = int(env.get('CONTENT_LENGTH', 0))
        except ValueError:
            request_body_size = 0

        request_body = env['wsgi.input'].read(request_body_size)
        request_json = json.loads(request_body.decode('utf-8'))
        prompt = request_json.get('prompt')

        if not prompt:
            response_body = json.dumps({'error': 'Missing prompt in request body'}).encode('utf-8')
            response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
            start_response('400 Bad Request', response_headers)
            return [response_body]

        data = self.get_service_response(prompt)
        response_body = self.format_response(data)
        
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]

    def handle_conversations_with_rag(self, env, start_response):
        """Handle POST requests to /api/rag_conversations with RAG functionality"""
        try:
            request_body_size = int(env.get('CONTENT_LENGTH', 0))
        except ValueError:
            request_body_size = 0

        request_body = env['wsgi.input'].read(request_body_size)
        request_json = json.loads(request_body.decode('utf-8'))
        prompt = request_json.get('prompt')

        if not prompt:
            response_body = json.dumps({'error': 'Missing prompt in request body'}).encode('utf-8')
            response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
            start_response('400 Bad Request', response_headers)
            return [response_body]

        data = self.get_service_response_with_rag(prompt)
        response_body = self.format_response(data)
        
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]

    def __http_200_ok(self, env, start_response):
        """Default handler for other routes"""
        try:
            request_body_size = int(env.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = env['wsgi.input'].read(request_body_size)
        request_json = json.loads(request_body.decode('utf-8'))
        prompt = request_json.get('prompt')

        data = self.get_service_response(prompt)
        response_body = self.format_response(data)
        
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]

    def __call__(self, env, start_response):
        method = env.get('REQUEST_METHOD').upper()
        path = env.get('PATH_INFO')

        if method != 'POST':
            return self.__http_415_notsupported(env, start_response)

        try:                
            handler = self.routes.get((method, path), self.__http_200_ok)
            return handler(env, start_response)
        except json.JSONDecodeError as e:
            response_body = json.dumps({'error': f"Invalid JSON: {e.msg}"}).encode('utf-8')
            response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
            start_response('400 Bad Request', response_headers)
            return [response_body]
        except Exception as e:
            # Log to stdout so it shows in GitHub Actions
            print("Exception occurred:")
            traceback.print_exc()

            # Return more detailed error response (would not do this in Production)
            error_response = json.dumps({'error': f"Internal Server Error: {str(e)}"}).encode('utf-8')
            response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(error_response)))]
            start_response('500 Internal Server Error', response_headers)
            return [error_response]