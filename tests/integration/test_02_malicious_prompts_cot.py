"""
    Usage:
    $ pytest -k test_02_malicious_prompts_cot -s

    Data used:
        1. Samples of compromised output (resulting from NVIDIA garak tests)
        2. Samples of prompt injection input

    Test cases (batches of 100+ runs each):
        2. Malicious prompt(s), zero-shot chain-of-thought prompt template

"""

import pytest
from test_utils import run_prompt_analysis_test
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService

@pytest.mark.mitigation
def test_02_malicious_prompts_cot(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return (service
                .with_chain_of_thought_guidelines())
    
    results = run_prompt_analysis_test(
        test_id=2,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )
        # assert result.mean < 0.5
        # at least 1 close match means the prompt is likely an injection attempt
        # assert result.max > 0.5
