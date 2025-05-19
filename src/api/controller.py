import logging
from flask import Flask, jsonify, request
from waitress import serve
from src.llm.llm import Phi3LanguageModel
from src.llm.llm_rag import Phi3LanguageModelWithRag

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return "Server is running", 200

@app.route('/api/conversations', methods=['POST'])
def get_llm_response():
    prompt = request.json['prompt']
    service = Phi3LanguageModel()
    response = service.invoke(user_input=prompt)
    return jsonify({'response': response}), 201

if __name__ == '__main__':
    logger = logging.Logger(name='Flask API', level=logging.DEBUG)
    print('test')
    logger.debug('running...')

    # Production mode with Waitress:
    serve(app, host='0.0.0.0', port=9999)