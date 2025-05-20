from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel

STATIC_PATH = Path(__file__).parent.absolute() / 'static'

app = FastAPI(
    title='Phi-3 Language Model API',
    description='HTTP API for interacting with Phi-3 Mini 4K language model'
)

class LanguageModelPrompt(BaseModel):
    prompt: str

class LanguageModelResponse(BaseModel):
    response: str


@app.get('/', response_model=str)
async def health_check():
    return 'success'
