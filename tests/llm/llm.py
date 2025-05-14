import argparse
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Llm:

    def __init__(self, model_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "phi3")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_id=model_path,
            device_map="cpu",  # Use available GPU
            trust_remote_code=True,  # If model requires custom code
        )

        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
        )

        # Create LangChain LLM
        self.hf_model = HuggingFacePipeline(pipeline=pipe)

    def get_response(self, input):
        # Use the model
        print(input)
        canned_input = "What is the capital of France?"
        print(canned_input)
        response = self.hf_model.invoke(canned_input)
        print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-m', '--model_path', type=str, required=False, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt input')
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', default=False, help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('--repetition_penalty', type=float, help='Repetition penalty to sample with')
    args = parser.parse_args()

    try:
        model_path = args.model_path
    except:
        model_path = None

    model = Llm(model_path)
    model.get_response(args.prompt)