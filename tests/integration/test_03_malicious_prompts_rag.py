"""
    Usage:
    $ pytest -k test_03_malicious_prompts_rag -s

    Data used:
        1. Samples of compromised output (resulting from NVIDIA garak tests)
        2. Samples of prompt injection input

    Test cases (batches of 100+ runs each):
        3. Malicious prompt(s), RAG with few-shot examples prompt template

"""

import pytest
from test_utils import run_prompt_analysis_test
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


@pytest.mark.mitigation
def test_03_malicious_prompts_rag(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    prompt_injection_sample_requests,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return (service
                .with_rag_context_guidelines())
    
    results = run_prompt_analysis_test(
        test_id=3,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=prompt_injection_sample_requests,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )
    
        # assert result.mean < 0.5
        # at least 1 close match means the prompt is likely an injection attempt
        # assert result.max > 0.5