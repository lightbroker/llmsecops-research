# TODO: business logic for REST API interaction w/ LLM via prompt input

import onnxruntime_genai as og
import argparse


default_model_path = './cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'

class Phi3LanguageModel:

    def __init__(self, model_path=None):
        # configure ONNX runtime
        model_path = default_model_path if model_path == None else model_path
        config = og.Config(model_path)
        config.clear_providers()
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()    
    

    def get_response(self, prompt_input):

        search_options = { 'max_length': 1024 }
        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        generator = og.Generator(self.model, params)

        # process prompt input and generate tokens
        chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        prompt = f'{chat_template.format(input=prompt_input)}'
        input_tokens = self.tokenizer.encode(prompt)
        generator.append_tokens(input_tokens)

        # generate output
        output = ''
        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                decoded = self.tokenizer_stream.decode(new_token)
                output = output + decoded
        except Exception as e:
            return f'{e}'
        return { 'response': output }


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

    model = Phi3LanguageModel(model_path)
    model.get_response(args.prompt)
