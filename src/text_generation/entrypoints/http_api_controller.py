import json
import traceback

from src.text_generation.services.logging.abstract_web_traffic_logging_service import AbstractWebTrafficLoggingService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService



class HttpApiController:
    def __init__(
            self, 
            logging_service: AbstractWebTrafficLoggingService,
            text_generation_response_service: AbstractTextGenerationCompletionService,
            rag_response_service: AbstractTextGenerationCompletionService,
            generated_text_guardrail_service: AbstractGeneratedTextGuardrailService
    ):
        self.logging_service = logging_service        
        self.text_generation_response_service = text_generation_response_service
        self.rag_response_service = rag_response_service
        self.generated_text_guardrail_service = generated_text_guardrail_service
        self.routes = {}
        self.register_routes()

    def debug_info(self, msg, *args, **kwargs):
        try:
            return self.original_info(msg, *args, **kwargs)
        except TypeError as e:
            print(f"Logging error with message: {repr(msg)}")
            print(f"Args: {args}")
            print(f"Kwargs: {kwargs}")
            raise e


    def register_routes(self):
        self.routes[('GET', '/')] = self.health_check
        self.routes[('POST', '/api/completions')] = self.handle_conversations
        self.routes[('POST', '/api/completions/rag-guided')] = self.handle_conversations_with_rag

    def format_response(self, data):
        response_data = {'response': data}
        try:
            response_body = json.dumps(response_data).encode('utf-8')
        except:
            response_body = json.dumps({'response': str(data)}).encode('utf-8')
        return response_body

    def health_check(self, env, start_response):
        response_body = self.format_response({ "success": True })
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]
    
    def handle_conversations(self, env, start_response):
        """POST /api/completions"""
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

        response_text = self.text_generation_response_service.invoke(user_prompt=prompt)
        score = self.generated_text_guardrail_service.process_generated_text(response_text)
        response_body = self.format_response(response_text)
        
        http_status_code = 200 # make enum
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response(f'{http_status_code} OK', response_headers)
        self.logging_service.log_request_response(request=prompt, response=response_text)
        return [response_body]

    def handle_conversations_with_rag(self, env, start_response):
        """POST /api/completions/rag-guided"""
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

        response_text = self.rag_response_service.invoke(user_prompt=prompt)
        score = self.generated_text_guardrail_service.process_generated_text(response_text)
        response_body = self.format_response(response_text)
        
        http_status_code = 200 # make enum
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response(f'{http_status_code} OK', response_headers)
        self.logging_service.log_request_response(request=prompt, response=response_text)
        return [response_body]

    def _http_200_ok(self, env, start_response):
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

        try:                
            handler = self.routes.get((method, path), self._http_200_ok)
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