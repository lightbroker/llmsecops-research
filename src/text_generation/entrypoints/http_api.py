# """
#     Usage:
#         $ uvicorn src.api.http_api:app --host 0.0.0.0 --port 9999
# """

# from fastapi import FastAPI
# from pathlib import Path
# from pydantic import BaseModel
# from src.llm.llm import Phi3LanguageModel


# STATIC_PATH = Path(__file__).parent.absolute() / 'static'

# app = FastAPI(
#     title='Phi-3 Language Model API',
#     description='HTTP API for interacting with Phi-3 Mini 4K language model'
# )

# class LanguageModelPrompt(BaseModel):
#     prompt: str

# class LanguageModelResponse(BaseModel):
#     response: str


# @app.get('/', response_model=str)
# async def health_check():
#     return 'success'


# @app.post('/api/conversations', response_model=LanguageModelResponse)
# async def get_llm_conversation_response(request: LanguageModelPrompt):
#     service = Phi3LanguageModel()
#     response = service.invoke(user_input=request.prompt)
#     return LanguageModelResponse(response=response)
