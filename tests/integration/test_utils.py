import inspect
import os
from typing import List, Callable
from src.text_generation.common.model_id import ModelId
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.services.logging.test_run_logging_service import TestRunLoggingService
from src.text_generation.services.nlp.abstract_semantic_similarity_service import AbstractSemanticSimilarityService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.services.nlp.text_generation_completion_service import TextGenerationCompletionService



def get_prompt_batch(prompts, batch_size=10, env_var='PROMPT_BATCH'):
    """
    Returns a batch of prompts based on the PROMPT_BATCH environment variable.
    Prints batch info for debugging.
    """
    batch_num = int(os.getenv(env_var, '1'))
    start_idx = (batch_num - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(prompts))
    prompt_subset = prompts[start_idx:end_idx]
    print(f"Running batch {batch_num}: prompts {start_idx+1}-{end_idx} ({len(prompt_subset)} prompts)")
    return prompt_subset


def run_prompt_analysis_test(
    test_id: int,
    model_id: ModelId,
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service: AbstractSemanticSimilarityService,
    prompts: List,
    comparison_texts: List,
    service_configurator: Callable,
    max_prompts: int = 100
) -> List[SemanticSimilarityResult]:
    """
    Common utility for running prompt analysis tests.
    
    Args:
        test_id: Unique identifier for the test
        text_generation_completion_service: Service for text generation
        semantic_similarity_service: Service for semantic analysis
        prompts: List of prompts to test
        comparison_texts: Reference texts for similarity comparison
        service_configurator: Function that configures the text generation service
        max_prompts: Maximum number of prompts to process (default: 100)
    
    Returns:
        List of SemanticSimilarityResult objects
    """
    semantic_similarity_service.use_comparison_texts(comparison_texts)
    print(f'using {len(prompts)} prompts for testing...')

    results = []

    for i, prompt in enumerate(prompts[:max_prompts], 1):
        # Configure the service using the provided configurator function
        configured_service: TextGenerationCompletionService = service_configurator(text_generation_completion_service)
        
        print(f'sending prompt {i} to LLM')
        completion_result: TextGenerationCompletionResult = configured_service.invoke(user_prompt=prompt, model_id=model_id)

        print(f'analyzing result')
        completion_result.finalize_completion_text()
        result: SemanticSimilarityResult = semantic_similarity_service.analyze(completion_result.final_completion_text)
        
        print(f'{i}/{len(prompts)} Max Score: {result.max}')
        print(f'{i}/{len(prompts)} Avg Score: {result.mean}')
        
        TestRunLoggingService(test_id=test_id, model_id=model_id).log_results(
            id=inspect.currentframe().f_back.f_code.co_name,
            text_generation_completion_result=completion_result,
            final_completion_text_score=result.max,
            is_rag_few_shot_enabled=text_generation_completion_service.is_rag_context_enabled(),
            is_cot_enabled=text_generation_completion_service.is_chain_of_thought_enabled(),
            is_reflexion_enabled=False,
            original_llm_config=completion_result.original_result.llm_config
        )
        
        results.append(result)
    
    return results