import abc

from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService


class LLMConfigurationIntrospectionService(
    AbstractLLMConfigurationIntrospectionService):
    # llm_configuration_introspection_service
    
    def get_config(llm_step):
        """
        Comprehensively extract all possible LLM configuration parameters
        from a HuggingFace pipeline step, checking all known locations.
        
        Returns:
            dict: All found configuration parameters that are JSON serializable
        """
        if not llm_step:
            return {}
        
        config = {}
        
        def safe_add_to_config(source_dict, source_name="unknown"):
            """Safely add items from a dict to config if they're serializable."""
            if not isinstance(source_dict, dict):
                return
                
            for key, value in source_dict.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    config[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                    config[key] = list(value)
                # Skip non-serializable objects
        
        # === LOCATION 1: Direct attributes on llm_step ===
        direct_llm_attrs = [
            # Generation parameters
            'temperature', 'top_p', 'top_k', 'max_new_tokens', 'max_length', 'min_length',
            'repetition_penalty', 'length_penalty', 'do_sample', 'early_stopping',
            'num_beams', 'num_beam_groups', 'diversity_penalty', 'typical_p',
            'epsilon_cutoff', 'eta_cutoff', 'exponential_decay_length_penalty',
            
            # Token IDs
            'pad_token_id', 'eos_token_id', 'bos_token_id', 'decoder_start_token_id',
            'forced_bos_token_id', 'forced_eos_token_id',
            
            # Model identifiers
            'model_id', 'model_name', 'model_path', 'model_type',
            
            # Task and device settings
            'task', 'device', 'device_map', 'torch_dtype',
            
            # Pipeline settings
            'batch_size', 'max_batch_size', 'return_full_text', 'clean_up_tokenization_spaces',
            'truncation', 'padding', 'add_special_tokens',
            
            # Performance settings
            'use_cache', 'cache_dir', 'revision', 'trust_remote_code',
            'low_cpu_mem_usage', 'load_in_8bit', 'load_in_4bit',
            
            # Quantization settings
            'quantization_config', 'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type',
            'bnb_4bit_use_double_quant',
            
            # Other generation settings
            'seed', 'guidance_scale', 'negative_prompt', 'num_images_per_prompt',
            'eta', 'generator', 'latents', 'prompt_embeds', 'negative_prompt_embeds',
            'cross_attention_kwargs', 'guidance_rescale', 'clip_skip',
            
            # Sampling parameters
            'top_a', 'tfs', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta',
            'penalty_alpha', 'use_mirostat_sampling',
            
            # Stop conditions
            'stop_sequences', 'stop_token_ids', 'stopping_criteria',
            
            # Memory and efficiency
            'offload_folder', 'cpu_offload', 'sequential_cpu_offload',
            'model_cpu_offload', 'disk_offload',
            
            # Framework specific
            'framework', 'use_fast', 'use_auth_token', 'subfolder',
        ]
        
        for attr in direct_llm_attrs:
            if hasattr(llm_step, attr):
                value = getattr(llm_step, attr)
                if isinstance(value, (str, int, float, bool, type(None))):
                    config[attr] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                    config[attr] = list(value)
        
        # === LOCATION 2: model_kwargs ===
        if hasattr(llm_step, 'model_kwargs') and llm_step.model_kwargs:
            safe_add_to_config(llm_step.model_kwargs, "model_kwargs")
        
        # === LOCATION 3: pipeline_kwargs ===
        if hasattr(llm_step, 'pipeline_kwargs') and llm_step.pipeline_kwargs:
            safe_add_to_config(llm_step.pipeline_kwargs, "pipeline_kwargs")
        
        # === LOCATION 4: Pipeline object and its attributes ===
        if hasattr(llm_step, 'pipeline') and llm_step.pipeline:
            pipeline = llm_step.pipeline
            
            # Direct pipeline attributes
            pipeline_attrs = [
                'temperature', 'top_p', 'top_k', 'max_new_tokens', 'max_length',
                'repetition_penalty', 'do_sample', 'pad_token_id', 'eos_token_id',
                'return_full_text', 'clean_up_tokenization_spaces', 'prefix',
                'handle_long_generation', 'batch_size'
            ]
            
            for attr in pipeline_attrs:
                if hasattr(pipeline, attr):
                    value = getattr(pipeline, attr)
                    if isinstance(value, (str, int, float, bool, type(None))):
                        config[attr] = value
            
            # Check pipeline._preprocess_params
            if hasattr(pipeline, '_preprocess_params'):
                safe_add_to_config(pipeline._preprocess_params, "_preprocess_params")
            
            # Check pipeline._forward_params
            if hasattr(pipeline, '_forward_params'):
                safe_add_to_config(pipeline._forward_params, "_forward_params")
            
            # Check pipeline._postprocess_params
            if hasattr(pipeline, '_postprocess_params'):
                safe_add_to_config(pipeline._postprocess_params, "_postprocess_params")
        
        # === LOCATION 5: Model's generation config ===
        if hasattr(llm_step, 'pipeline') and llm_step.pipeline:
            pipeline = llm_step.pipeline
            
            # Try to access generation config through model
            try:
                if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'generation_config'):
                    gen_config = pipeline.model.generation_config
                    if hasattr(gen_config, 'to_dict'):
                        gen_dict = gen_config.to_dict()
                        safe_add_to_config(gen_dict, "generation_config")
                    elif hasattr(gen_config, '__dict__'):
                        safe_add_to_config(gen_config.__dict__, "generation_config_dict")
            except Exception as e:
                # Silently continue if generation config access fails
                pass
            
            # Try to access config through model.config
            try:
                if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'config'):
                    model_config = pipeline.model.config
                    if hasattr(model_config, 'to_dict'):
                        model_config_dict = model_config.to_dict()
                        # Only extract generation-related config items
                        generation_keys = [
                            'max_length', 'max_new_tokens', 'min_length', 'do_sample',
                            'temperature', 'top_k', 'top_p', 'repetition_penalty',
                            'length_penalty', 'num_beams', 'early_stopping',
                            'pad_token_id', 'eos_token_id', 'bos_token_id'
                        ]
                        for key in generation_keys:
                            if key in model_config_dict:
                                value = model_config_dict[key]
                                if isinstance(value, (str, int, float, bool, type(None))):
                                    config[key] = value
            except Exception as e:
                # Silently continue if model config access fails
                pass
        
        # === LOCATION 6: Tokenizer config ===
        if hasattr(llm_step, 'pipeline') and llm_step.pipeline:
            try:
                if hasattr(llm_step.pipeline, 'tokenizer'):
                    tokenizer = llm_step.pipeline.tokenizer
                    tokenizer_attrs = [
                        'pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id',
                        'sep_token_id', 'cls_token_id', 'mask_token_id',
                        'padding_side', 'truncation_side', 'model_max_length'
                    ]
                    
                    for attr in tokenizer_attrs:
                        if hasattr(tokenizer, attr):
                            value = getattr(tokenizer, attr)
                            if isinstance(value, (str, int, float, bool, type(None))):
                                config[f"tokenizer_{attr}"] = value
            except Exception as e:
                # Silently continue if tokenizer access fails
                pass
        
        # === LOCATION 7: Try model_dump with filtering ===
        try:
            full_dump = llm_step.model_dump()
            if isinstance(full_dump, dict):
                # List of keys we definitely want to try to extract
                priority_keys = [
                    'temperature', 'top_p', 'top_k', 'max_new_tokens', 'max_length',
                    'repetition_penalty', 'do_sample', 'pad_token_id', 'eos_token_id',
                    'model_id', 'task', 'device', 'batch_size', 'return_full_text',
                    'model_kwargs', 'pipeline_kwargs'
                ]
                
                for key in priority_keys:
                    if key in full_dump:
                        value = full_dump[key]
                        if isinstance(value, (str, int, float, bool, type(None))):
                            config[key] = value
                        elif isinstance(value, dict):
                            # If it's a nested dict, try to extract from it
                            safe_add_to_config(value, f"model_dump_{key}")
        except Exception as e:
            # model_dump might fail due to non-serializable objects
            pass
        
        # === LOCATION 8: Check for any additional generation parameters ===
        # Look for any attributes ending in common parameter suffixes
        if hasattr(llm_step, '__dict__'):
            for attr_name, attr_value in llm_step.__dict__.items():
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    # Add if it looks like a generation parameter
                    if any(suffix in attr_name.lower() for suffix in [
                        'temperature', 'top_', 'max_', 'min_', 'penalty', 'token_id',
                        'sample', 'beam', 'length', 'config', 'param'
                    ]):
                        config[attr_name] = attr_value
        
        # === CLEANUP: Remove duplicates and None values (optional) ===
        # Remove None values if desired
        # config = {k: v for k, v in config.items() if v is not None}
        
        return config


    # Helper function to pretty print the config for debugging
    def print_llm_config_debug(llm_step):
        """Debug helper to print all found configuration in organized format."""
        config = extract_all_llm_config(llm_step)
        
        if not config:
            print("No LLM configuration found")
            return config
        
        print("=== EXTRACTED LLM CONFIGURATION ===")
        
        # Group by category for better readability
        categories = {
            'Generation Parameters': [
                'temperature', 'top_p', 'top_k', 'max_new_tokens', 'max_length', 'min_length',
                'repetition_penalty', 'length_penalty', 'do_sample', 'num_beams', 'early_stopping'
            ],
            'Token IDs': [
                'pad_token_id', 'eos_token_id', 'bos_token_id', 'decoder_start_token_id'
            ],
            'Model Info': [
                'model_id', 'model_name', 'model_path', 'model_type', 'task'
            ],
            'Device & Performance': [
                'device', 'device_map', 'batch_size', 'use_cache', 'torch_dtype'
            ],
            'Pipeline Settings': [
                'return_full_text', 'clean_up_tokenization_spaces', 'truncation', 'padding'
            ]
        }
        
        for category, keys in categories.items():
            found_in_category = {k: v for k, v in config.items() if k in keys}
            if found_in_category:
                print(f"\n{category}:")
                for key, value in found_in_category.items():
                    print(f"  {key}: {value}")
        
        # Print any remaining parameters
        categorized_keys = set()
        for keys in categories.values():
            categorized_keys.update(keys)
        
        remaining = {k: v for k, v in config.items() if k not in categorized_keys}
        if remaining:
            print(f"\nOther Parameters:")
            for key, value in remaining.items():
                print(f"  {key}: {value}")
        
        print(f"\nTotal parameters found: {len(config)}")
        return config