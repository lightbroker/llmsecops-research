import logging
from flask import Flask, jsonify, request
from src.llm.llm import Phi3LanguageModel
from src.llm.llm_rag import Phi3LanguageModelWithRag

app = Flask(__name__)

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
    app.run(debug=True, port=9998)