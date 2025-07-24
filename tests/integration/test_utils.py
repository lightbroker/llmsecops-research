import inspect
from typing import List, Callable
from src.text_generation.domain.semantic_similarity_result import SemanticSimilarityResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult
from src.text_generation.services.logging.test_run_logging_service import TestRunLoggingService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


def run_prompt_analysis_test(
    test_id: int,
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
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
    results = []
    
    for i, prompt in enumerate(prompts[:max_prompts], 1):
        # Configure the service using the provided configurator function
        configured_service = service_configurator(text_generation_completion_service)
        
        completion_result: TextGenerationCompletionResult = configured_service.invoke(user_prompt=prompt)
        result: SemanticSimilarityResult = semantic_similarity_service.analyze(completion_result.final)
        
        print(f'{i}/{len(prompts)} Max Score: {result.max}')
        print(f'{i}/{len(prompts)} Avg Score: {result.mean}')
        
        TestRunLoggingService(test_id=test_id).log_results(
            id=inspect.currentframe().f_back.f_code.co_name,
            prompt=prompt,
            completion=completion_result.original_completion,
            is_rag_few_shot_enabled=text_generation_completion_service.is_rag_context_enabled(),
            is_cot_enabled=text_generation_completion_service.is_chain_of_thought_enabled(),
            is_reflexion_enabled=text_generation_completion_service.is_reflexion_enabled(),
            scores=result.scores,
            mean=result.mean,
            max=result.max
        )
        
        results.append(result)
    
    return results