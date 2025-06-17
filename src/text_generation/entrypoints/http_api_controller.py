import json
import traceback

from src.text_generation.services.language_models.text_generation_response_service import TextGenerationResponseService
from src.text_generation.services.language_models.retrieval_augmented_generation_response_service import RetrievalAugmentedGenerationResponseService
from src.text_generation.services.logging.file_logging_service import FileLoggingService

class HttpApiController:
    def __init__(
            self, 
            logging_service: FileLoggingService,
            text_generation_response_service: TextGenerationResponseService,
            rag_response_service: RetrievalAugmentedGenerationResponseService
    ):
        self.logger = logging_service.logger
        
        # TODO: temp debug
        self.original_info = self.logger.info
        self.logger.info = self.debug_info
        
        self.text_generation_response_service = text_generation_response_service
        self.rag_response_service = rag_response_service
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
        """Register all API routes"""
        self.routes[('GET', '/')] = self.health_check
        self.routes[('POST', '/api/conversations')] = self.handle_conversations
        self.routes[('POST', '/api/rag_conversations')] = self.handle_conversations_with_rag

    def __http_415_notsupported(self, env, start_response):
        response_headers = [('Content-Type', 'application/json')]
        start_response('415 Unsupported Media Type', response_headers)
        return [json.dumps({'error': 'Unsupported Content-Type'}).encode('utf-8')]

    def format_response(self, data):
        """Format response data as JSON with 'response' key"""
        response_data = {'response': data}
        try:
            response_body = json.dumps(response_data).encode('utf-8')
        except:
            # If serialization fails, convert data to string first
            response_body = json.dumps({'response': str(data)}).encode('utf-8')
        return response_body

    def health_check(self, env, start_response):
        response_body = self.format_response({ "success": True })
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response('200 OK', response_headers)    
        return [response_body]
    
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

        response_text = self.text_generation_response_service.invoke(user_prompt=prompt)
        response_body = self.format_response(response_text)
        
        http_status_code = 200 # make enum
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response(f'{http_status_code} OK', response_headers)
        self.logger.info('non-RAG response', request_body, http_status_code, response_body)
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

        response_text = self.rag_response_service.invoke(user_prompt=prompt)
        response_body = self.format_response(response_text)
        
        http_status_code = 200 # make enum
        response_headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(response_body)))]
        start_response(f'{http_status_code} OK', response_headers)
        self.logger.info('RAG response', request_body, http_status_code, response_body)
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