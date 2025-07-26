import json
import os
from typing import Dict, List
from src.text_generation.ports.abstract_prompt_injection_example_repository import AbstractPromptInjectionExampleRepository


class PromptInjectionExampleRepository(AbstractPromptInjectionExampleRepository):
    """Repository containing prompt injection examples."""

    def __init__(self):
        super().__init__()
        self.templates_dir = os.environ.get('INJECTION_DATA_DIR')

    @property
    def _examples(self) -> List[Dict[str, str]]:
        """Get the list of prompt injection examples."""

        base_path = os.environ.get('INJECTION_DATA_DIR', '.')    
        json_filename = 'process_random_requests_and_responses.output.545_count.json'
        file_path = os.path.join(base_path, json_filename)
        result = []
    
        try:
            print(f'opening {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Handle both single object and array of objects
            if isinstance(data, dict):
                data = [data]
            elif isinstance(data, list):
                pass
            else:
                print("Error: JSON file should contain an object or array of objects")
                return []
            
            print(f'data length: {len(data)}')
            
            for item in data:
                if isinstance(item, dict):
                    # Extract prompt and completion keys, ignore others
                    extracted = {}

                    prompt_keys = ['prompt_injection_prompt']
                    for key in prompt_keys:
                        if key in item:
                            extracted['prompt_injection_prompt'] = item[key].replace('{', '').replace('}', '')
                            break

                    completion_keys = ['prompt_injection_completion']
                    for key in completion_keys:
                        if key in item:
                            extracted['prompt_injection_completion'] = item[key].replace('{', '').replace('}', '')
                            break
                    
                    if 'prompt_injection_prompt' in extracted and 'prompt_injection_completion' in extracted:
                        result.append(extracted)
        
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
        except Exception as e:
            print(f"Error: {e}")
        
        return result


    def get_all(self) -> List[Dict[str, str]]:
        """Get all prompt injection examples."""
        return self._examples