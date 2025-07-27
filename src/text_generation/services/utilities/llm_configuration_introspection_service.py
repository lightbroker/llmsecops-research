import abc

from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService


class LLMConfigurationIntrospectionService(
    AbstractLLMConfigurationIntrospectionService):
    # llm_configuration_introspection_service


    def get_config(self, lcel_chain, max_depth=10):
        """
        Comprehensively extract all possible LLM configuration parameters
        from a LangChain LCEL chain object, creating a multilayered dict structure
        that preserves the chain hierarchy.
        
        Args:
            lcel_chain: A LangChain LCEL chain object (Runnable)
            max_depth: Maximum recursion depth to prevent infinite loops
        
        Returns:
            dict: Nested dictionary with full chain structure and all config parameters
        """
        if not lcel_chain or max_depth <= 0:
            return {}
        
        def is_serializable(value):
            """Check if a value is JSON serializable."""
            return isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
        
        def safe_serialize(value):
            """Safely serialize a value, converting non-serializable objects to strings."""
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [safe_serialize(item) for item in value]
            elif isinstance(value, dict):
                return {k: safe_serialize(v) for k, v in value.items() if k != '_type'}
            else:
                # Convert objects to string representation, but filter out some noise
                str_repr = str(value)
                if any(noise in str_repr for noise in ['<bound method', '<function', 'object at 0x']):
                    return f"<{type(value).__name__}>"
                return str_repr
        
        def extract_from_object(obj, path="root", visited=None, current_depth=0):
            """
            Recursively extract configuration from any object, building a nested structure.
            """
            if visited is None:
                visited = set()
            
            if current_depth >= max_depth or id(obj) in visited:
                return {}
            
            visited.add(id(obj))
            result = {"_type": type(obj).__name__, "_path": path}
            
            # === COMPREHENSIVE ATTRIBUTE EXTRACTION ===
            
            # All possible LLM and chain configuration attributes
            all_config_attrs = [
                # Core generation parameters
                'temperature', 'top_p', 'top_k', 'max_tokens', 'max_new_tokens', 'max_length',
                'min_length', 'repetition_penalty', 'frequency_penalty', 'presence_penalty',
                'length_penalty', 'do_sample', 'early_stopping', 'num_beams', 'num_beam_groups',
                'diversity_penalty', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'seed',
                'stop', 'stop_sequences', 'suffix', 'logit_bias', 'user', 'n', 'best_of',
                'logprobs', 'echo', 'response_format', 'tool_choice', 'parallel_tool_calls',
                
                # Model and API configuration
                'model', 'model_name', 'model_id', 'model_path', 'model_type', 'engine',
                'deployment_name', 'deployment_id', 'model_version', 'model_revision',
                'api_key', 'api_base', 'api_version', 'api_type', 'organization', 'base_url',
                'endpoint', 'region', 'project_id', 'project', 'location', 'credentials',
                
                # Provider-specific keys
                'openai_api_key', 'openai_organization', 'openai_api_base', 'openai_proxy',
                'anthropic_api_key', 'anthropic_api_url', 'max_tokens_to_sample',
                'cohere_api_key', 'huggingfacehub_api_token', 'repo_id', 'task',
                'google_api_key', 'vertex_ai_model', 'azure_endpoint', 'azure_deployment',
                'azure_api_version', 'azure_api_key', 'replicate_api_token',
                'together_api_key', 'fireworks_api_key', 'groq_api_key', 'mistral_api_key',
                
                # Request and performance settings
                'max_retries', 'request_timeout', 'timeout', 'streaming', 'chunk_size',
                'max_concurrent_requests', 'rate_limit', 'batch_size', 'max_batch_size',
                'use_cache', 'cache_dir', 'cache_size', 'device', 'device_map', 'torch_dtype',
                'load_in_8bit', 'load_in_4bit', 'trust_remote_code', 'revision',
                
                # Token handling
                'pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id',
                'sep_token_id', 'cls_token_id', 'mask_token_id', 'decoder_start_token_id',
                'forced_bos_token_id', 'forced_eos_token_id',
                
                # Chain-specific attributes
                'verbose', 'name', 'tags', 'metadata', 'callbacks', 'memory', 'memory_key',
                'return_messages', 'input_key', 'output_key', 'prompt', 'llm_chain',
                'combine_documents_chain', 'question_generator', 'retriever',
                
                # Pipeline and processing
                'return_full_text', 'clean_up_tokenization_spaces', 'truncation', 'padding',
                'add_special_tokens', 'handle_long_generation', 'prefix',
                
                # Advanced parameters
                'penalty_alpha', 'use_mirostat_sampling', 'mirostat_mode', 'mirostat_tau',
                'mirostat_eta', 'tfs', 'top_a', 'k', 'p', 'include_stop_str_in_output',
                'ignore_eos', 'skip_special_tokens', 'spaces_between_special_tokens',
            ]
            
            # === PRIORITY: Extract critical generation parameters first ===
            critical_params = ['temperature', 'top_k', 'top_p', 'max_length', 'max_new_tokens', 
                            'max_tokens', 'repetition_penalty', 'do_sample', 'num_beams']
            
            for param in critical_params:
                # Check multiple possible locations for each critical parameter
                found_value = None
                locations_to_check = [
                    # Direct attribute
                    (lambda: getattr(obj, param) if hasattr(obj, param) else None, f"direct.{param}"),
                    
                    # In various config containers
                    (lambda: getattr(obj, 'model_kwargs', {}).get(param) if hasattr(obj, 'model_kwargs') else None, f"model_kwargs.{param}"),
                    (lambda: getattr(obj, 'pipeline_kwargs', {}).get(param) if hasattr(obj, 'pipeline_kwargs') else None, f"pipeline_kwargs.{param}"),
                    (lambda: getattr(obj, 'generation_config', {}).get(param) if hasattr(obj, 'generation_config') else None, f"generation_config.{param}"),
                    (lambda: getattr(obj, 'kwargs', {}).get(param) if hasattr(obj, 'kwargs') else None, f"kwargs.{param}"),
                    (lambda: getattr(obj, '_config', {}).get(param) if hasattr(obj, '_config') else None, f"_config.{param}"),
                    
                    # In nested pipeline object
                    (lambda: getattr(getattr(obj, 'pipeline', None), param, None) if hasattr(obj, 'pipeline') else None, f"pipeline.{param}"),
                    (lambda: getattr(getattr(obj, 'pipeline', None), '_preprocess_params', {}).get(param) if hasattr(obj, 'pipeline') else None, f"pipeline._preprocess_params.{param}"),
                    (lambda: getattr(getattr(obj, 'pipeline', None), '_forward_params', {}).get(param) if hasattr(obj, 'pipeline') else None, f"pipeline._forward_params.{param}"),
                    (lambda: getattr(getattr(obj, 'pipeline', None), '_postprocess_params', {}).get(param) if hasattr(obj, 'pipeline') else None, f"pipeline._postprocess_params.{param}"),
                    
                    # In model's generation config
                    (lambda: getattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config', None).__dict__.get(param) if hasattr(obj, 'pipeline') and hasattr(getattr(obj, 'pipeline', None), 'model') and hasattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config') else None, f"pipeline.model.generation_config.{param}"),
                    
                    # Try generation_config.to_dict()
                    (lambda: getattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config', None).to_dict().get(param) if hasattr(obj, 'pipeline') and hasattr(getattr(obj, 'pipeline', None), 'model') and hasattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config') and hasattr(getattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config', None), 'to_dict') else None, f"pipeline.model.generation_config.to_dict().{param}"),
                    
                    # Check in model config
                    (lambda: getattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'config', None).__dict__.get(param) if hasattr(obj, 'pipeline') and hasattr(getattr(obj, 'pipeline', None), 'model') and hasattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'config') else None, f"pipeline.model.config.{param}"),
                    
                    # Check bound parameters
                    (lambda: getattr(obj, 'bound', {}).get(param) if hasattr(obj, 'bound') else None, f"bound.{param}"),
                    
                    # Check __dict__ directly
                    (lambda: obj.__dict__.get(param) if hasattr(obj, '__dict__') else None, f"__dict__.{param}"),
                ]
                
                for getter, location in locations_to_check:
                    try:
                        value = getter()
                        if value is not None:
                            found_value = value
                            result[param] = safe_serialize(value)
                            result[f"{param}_source"] = location  # Track where we found it
                            break
                    except Exception:
                        continue
                
                # If still not found, do a deeper search in __dict__
                if found_value is None and hasattr(obj, '__dict__'):
                    for key, value in obj.__dict__.items():
                        if param in key.lower() and value is not None:
                            result[f"{param}_from_{key}"] = safe_serialize(value)
                            break
            
            # Extract all other attributes
            for attr in all_config_attrs:
                if attr not in critical_params and hasattr(obj, attr):
                    try:
                        value = getattr(obj, attr)
                        if value is not None:
                            result[attr] = safe_serialize(value)
                    except Exception as e:
                        result[f"{attr}_error"] = str(e)
            
            # === EXTRACT FROM COMMON CONFIG CONTAINERS ===
            config_containers = [
                'kwargs', 'model_kwargs', 'pipeline_kwargs', 'llm_kwargs', 'generation_config',
                'config', '_config', 'params', '_params', 'bound', 'default_params',
                '_preprocess_params', '_forward_params', '_postprocess_params'
            ]
            
            for container_name in config_containers:
                if hasattr(obj, container_name):
                    try:
                        container = getattr(obj, container_name)
                        if isinstance(container, dict) and container:
                            result[container_name] = safe_serialize(container)
                    except Exception:
                        pass
            
            # === EXTRACT FROM __DICT__ ===
            if hasattr(obj, '__dict__'):
                obj_dict = {}
                for key, value in obj.__dict__.items():
                    # Skip private/internal attributes and known non-config items
                    if (not key.startswith('_') or key in ['_config', '_params']) and \
                    key not in ['callbacks'] and \
                    not callable(value):
                        try:
                            if is_serializable(value) or isinstance(value, (dict, list)):
                                obj_dict[key] = safe_serialize(value)
                            elif hasattr(value, '__dict__') or hasattr(value, 'dict'):
                                # This might be a nested config object
                                nested_config = extract_from_object(
                                    value, f"{path}.{key}", visited.copy(), current_depth + 1
                                )
                                if nested_config and len(nested_config) > 2:  # More than just _type and _path
                                    obj_dict[key] = nested_config
                        except Exception:
                            pass
                
                if obj_dict:
                    result['_attributes'] = obj_dict
            
            # === HANDLE SPECIFIC CHAIN STRUCTURES ===
            
            # Sequential chains (RunnableSequence)
            if hasattr(obj, 'steps') and obj.steps:
                steps_config = {}
                for i, step in enumerate(obj.steps):
                    step_config = extract_from_object(
                        step, f"{path}.steps[{i}]", visited.copy(), current_depth + 1
                    )
                    if step_config:
                        steps_config[f"step_{i}"] = step_config
                if steps_config:
                    result['steps'] = steps_config
            
            # Parallel chains (RunnableParallel)
            if hasattr(obj, 'mapping') and isinstance(obj.mapping, dict):
                mapping_config = {}
                for key, component in obj.mapping.items():
                    comp_config = extract_from_object(
                        component, f"{path}.mapping[{key}]", visited.copy(), current_depth + 1
                    )
                    if comp_config:
                        mapping_config[key] = comp_config
                if mapping_config:
                    result['mapping'] = mapping_config
            
            # Conditional chains (RunnableBranch)
            if hasattr(obj, 'branches') and obj.branches:
                branches_config = {}
                for i, (condition, branch) in enumerate(obj.branches):
                    branch_config = extract_from_object(
                        branch, f"{path}.branches[{i}]", visited.copy(), current_depth + 1
                    )
                    if branch_config:
                        branches_config[f"branch_{i}"] = branch_config
                if branches_config:
                    result['branches'] = branches_config
            
            if hasattr(obj, 'default') and obj.default:
                default_config = extract_from_object(
                    obj.default, f"{path}.default", visited.copy(), current_depth + 1
                )
                if default_config:
                    result['default'] = default_config
            
            # Chain components
            component_attrs = [
                'llm', 'model', 'language_model', 'chat_model', 'completion_model',
                'first', 'last', 'middle', 'chain', 'inner_chain', 'base_chain',
                'retrieval_chain', 'combine_documents_chain', 'question_generator',
                'memory', 'retriever', 'prompt', 'output_parser', 'parser'
            ]
            
            for comp_attr in component_attrs:
                if hasattr(obj, comp_attr):
                    try:
                        component = getattr(obj, comp_attr)
                        if component and not callable(component):
                            if isinstance(component, list):
                                comp_configs = {}
                                for i, item in enumerate(component):
                                    item_config = extract_from_object(
                                        item, f"{path}.{comp_attr}[{i}]", visited.copy(), current_depth + 1
                                    )
                                    if item_config:
                                        comp_configs[f"{comp_attr}_{i}"] = item_config
                                if comp_configs:
                                    result[comp_attr] = comp_configs
                            else:
                                comp_config = extract_from_object(
                                    component, f"{path}.{comp_attr}", visited.copy(), current_depth + 1
                                )
                                if comp_config and len(comp_config) > 2:
                                    result[comp_attr] = comp_config
                    except Exception:
                        pass
            
            # Try model.dict() or similar serialization methods
            for method_name in ['dict', 'model_dump', 'to_dict', 'serialize']:
                if hasattr(obj, method_name):
                    try:
                        method = getattr(obj, method_name)
                        if callable(method):
                            serialized = method()
                            if isinstance(serialized, dict) and serialized:
                                result[f'_{method_name}'] = safe_serialize(serialized)
                            break  # Only use the first successful method
                    except Exception:
                        pass
            
            return result
        
        # Start extraction from the root chain
        return extract_from_object(lcel_chain)


    def print_nested_config(self, config, indent=0, max_items_per_level=50):
        """
        Pretty print the nested configuration structure.
        """
        if not isinstance(config, dict):
            print("  " * indent + str(config))
            return
        
        items_shown = 0
        for key, value in config.items():
            if items_shown >= max_items_per_level:
                print("  " * indent + f"... ({len(config) - items_shown} more items)")
                break
                
            if key.startswith('_') and key not in ['_type', '_path']:
                continue  # Skip most internal fields in main display
                
            print("  " * indent + f"{key}:")
            
            if isinstance(value, dict):
                if key == '_attributes' and indent > 0:
                    # Flatten attributes for readability
                    attr_count = 0
                    for attr_key, attr_val in value.items():
                        if attr_count >= 10:  # Limit attribute display
                            print("  " * (indent + 1) + f"... ({len(value) - attr_count} more attributes)")
                            break
                        if not isinstance(attr_val, dict):
                            print("  " * (indent + 1) + f"{attr_key}: {attr_val}")
                            attr_count += 1
                else:
                    self.print_nested_config(value, indent + 1, max_items_per_level)
            else:
                print("  " * (indent + 1) + str(value))
            
            items_shown += 1


    def extract_flattened_config(self, lcel_chain):
        """
        Extract and flatten all configuration into a single-level dictionary
        with dotted paths showing the source hierarchy.
        """
        nested = self.extract_all_llm_config(lcel_chain)
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                if k.startswith('_'):
                    continue  # Skip metadata
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        return flatten_dict(nested)


    def find_critical_generation_params(self, lcel_chain):
        """
        Specifically hunt for the most critical generation parameters that are often missing.
        Returns a focused dict with just the essential params and where they were found.
        """
        critical_params = {
            'temperature': None,
            'top_k': None, 
            'top_p': None,
            'max_length': None,
            'max_new_tokens': None,
            'max_tokens': None,
            'repetition_penalty': None,
            'do_sample': None
        }
        
        def deep_search_for_param(obj, param_name, visited=None, path=""):
            if visited is None:
                visited = set()
            if id(obj) in visited:
                return None
            visited.add(id(obj))
            
            # All possible locations to check
            search_locations = [
                # Direct attribute
                lambda: getattr(obj, param_name, None),
                # In common config dicts
                lambda: getattr(obj, 'model_kwargs', {}).get(param_name),
                lambda: getattr(obj, 'pipeline_kwargs', {}).get(param_name),
                lambda: getattr(obj, 'kwargs', {}).get(param_name),
                lambda: getattr(obj, 'generation_config', {}).get(param_name),
                lambda: getattr(obj, '_config', {}).get(param_name),
                lambda: getattr(obj, 'bound', {}).get(param_name),
                # In pipeline
                lambda: getattr(getattr(obj, 'pipeline', None), param_name, None),
                # In pipeline config dicts
                lambda: getattr(getattr(obj, 'pipeline', None), '_preprocess_params', {}).get(param_name) if hasattr(obj, 'pipeline') else None,
                lambda: getattr(getattr(obj, 'pipeline', None), '_forward_params', {}).get(param_name) if hasattr(obj, 'pipeline') else None,
                lambda: getattr(getattr(obj, 'pipeline', None), '_postprocess_params', {}).get(param_name) if hasattr(obj, 'pipeline') else None,
                # In model generation config
                lambda: getattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config', None).__dict__.get(param_name) if hasattr(obj, 'pipeline') and hasattr(getattr(obj, 'pipeline', None), 'model', None) and hasattr(getattr(getattr(obj, 'pipeline', None), 'model', None), 'generation_config') else None,
            ]
            
            for search_func in search_locations:
                try:
                    value = search_func()
                    if value is not None:
                        return {"value": value, "location": f"{path} -> {search_func.__name__}"}
                except:
                    continue
            
            # Recurse into sub-objects
            if hasattr(obj, 'steps'):
                for i, step in enumerate(obj.steps):
                    result = deep_search_for_param(step, param_name, visited.copy(), f"{path}.steps[{i}]")
                    if result:
                        return result
            
            if hasattr(obj, 'mapping') and isinstance(obj.mapping, dict):
                for key, component in obj.mapping.items():
                    result = deep_search_for_param(component, param_name, visited.copy(), f"{path}.mapping[{key}]")
                    if result:
                        return result
            
            # Check common component attributes
            for attr_name in ['llm', 'model', 'pipeline', 'chain']:
                if hasattr(obj, attr_name):
                    component = getattr(obj, attr_name)
                    if component:
                        result = deep_search_for_param(component, param_name, visited.copy(), f"{path}.{attr_name}")
                        if result:
                            return result
            
            return None
        
        print("=== HUNTING FOR CRITICAL GENERATION PARAMETERS ===")
        for param in critical_params:
            result = deep_search_for_param(lcel_chain, param)
            if result:
                critical_params[param] = result
                print(f"✓ Found {param}: {result['value']} (from: {result['location']})")
            else:
                print(f"✗ Missing {param}")
        
        return critical_params


    def print_llm_config_debug(self, lcel_chain):
        """
        Debug helper that shows both nested and flattened views of the configuration.
        """
        print("=== CRITICAL PARAMETERS SEARCH ===")
        critical = self.find_critical_generation_params(lcel_chain)
        
        print("\n" + "="*50)
        print("=== NESTED LCEL CHAIN CONFIGURATION ===")
        nested_config = self.extract_all_llm_config(lcel_chain)
        self.print_nested_config(nested_config)
        
        print("\n" + "="*50)
        print("=== FLATTENED CONFIGURATION ===")
        flattened = self.extract_flattened_config(lcel_chain)
        
        if not flattened:
            print("No configuration parameters found")
            return nested_config
        
        # Group by category with priority for generation params
        categories = {
            '🔥 CRITICAL Generation Parameters': [],
            'Other Generation Parameters': [],
            'Model Configuration': [],
            'API Settings': [],
            'Chain Structure': [],
            'Other': []
        }
        
        critical_param_names = ['temperature', 'top_k', 'top_p', 'max_length', 'max_new_tokens', 'max_tokens']
        
        for key, value in flattened.items():
            categorized = False
            
            # Check if it's a critical parameter
            if any(param in key.lower() for param in critical_param_names):
                categories['🔥 CRITICAL Generation Parameters'].append((key, value))
                categorized = True
            elif any(param in key.lower() for param in ['penalty', 'sample', 'beam', 'length']):
                categories['Other Generation Parameters'].append((key, value))
                categorized = True
            elif any(param in key.lower() for param in ['model', 'engine', 'deployment']):
                categories['Model Configuration'].append((key, value))
                categorized = True
            elif any(param in key.lower() for param in ['api', 'key', 'endpoint', 'url', 'timeout']):
                categories['API Settings'].append((key, value))
                categorized = True
            elif any(param in key.lower() for param in ['step', 'chain', 'mapping', 'branch']):
                categories['Chain Structure'].append((key, value))
                categorized = True
            
            if not categorized:
                categories['Other'].append((key, value))
        
        for category, items in categories.items():
            if items:
                print(f"\n{category}:")
                for key, value in items:
                    print(f"  {key}: {value}")
        
        print(f"\nTotal parameters found: {len(flattened)}")
        return nested_config


    # Example usage with detailed iteration
    def iterate_chain_components(self, lcel_chain):
        """
        Example function showing how to iterate through all chain components
        and extract configuration from each.
        """
        print("=== ITERATING THROUGH CHAIN COMPONENTS ===")
        
        def visit_component(component, path="root", depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            
            print("  " * depth + f"Visiting: {path} ({type(component).__name__})")
            
            # Extract config from this component
            config = {}
            
            # Check for common LLM attributes
            llm_attrs = ['temperature', 'top_p', 'model', 'model_id', 'max_tokens', 'api_key']
            for attr in llm_attrs:
                if hasattr(component, attr):
                    value = getattr(component, attr)
                    if value is not None:
                        config[attr] = value
            
            if config:
                print("  " * depth + f"  Config found: {config}")
            
            # Recurse into sub-components
            if hasattr(component, 'steps'):
                for i, step in enumerate(component.steps):
                    visit_component(step, f"{path}.steps[{i}]", depth + 1)
            
            if hasattr(component, 'mapping') and isinstance(component.mapping, dict):
                for key, subcomp in component.mapping.items():
                    visit_component(subcomp, f"{path}.mapping[{key}]", depth + 1)
            
            if hasattr(component, 'llm') and component.llm:
                visit_component(component.llm, f"{path}.llm", depth + 1)
            
            if hasattr(component, 'model') and component.model:
                visit_component(component.model, f"{path}.model", depth + 1)
        
        visit_component(lcel_chain)


    # Complete usage example
    def example_usage(self):
        """
        Complete example showing all extraction methods.
        """
        print("=== LANGCHAIN LCEL CONFIG EXTRACTOR USAGE ===\n")
        
        print("1. NESTED STRUCTURE EXTRACTION:")
        print("   nested_config = extract_all_llm_config(chain)")
        print("   # Returns: Full nested dict preserving chain hierarchy")
        
        print("\n2. FLATTENED EXTRACTION:")
        print("   flat_config = extract_flattened_config(chain)")
        print("   # Returns: Single-level dict with dotted path keys")
        
        print("\n3. DEBUG OUTPUT:")
        print("   print_llm_config_debug(chain)")
        print("   # Prints: Both nested and categorized flat views")
        
        print("\n4. COMPONENT ITERATION:")
        print("   iterate_chain_components(chain)")
        print("   # Shows: Step-by-step traversal of all components")
        
        print("\nExample output structure:")
        example_structure = {
            "_type": "RunnableSequence",
            "steps": {
                "step_0": {
                    "_type": "ChatPromptTemplate",
                    "template": "You are a helpful assistant"
                },
                "step_1": {
                    "_type": "ChatOpenAI",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "openai_api_key": "sk-...",
                    "_attributes": {
                        "streaming": False,
                        "verbose": False
                    }
                }
            }
        }
        
        self.print_nested_config(example_structure)

