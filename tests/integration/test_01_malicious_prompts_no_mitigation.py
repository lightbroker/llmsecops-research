"""
    Usage:
    $ pytest -k test_01_malicious_prompts_no_mitigation -s --disable-warnings
"""

import pytest
from test_utils import run_prompt_analysis_test
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


@pytest.mark.mitigation
def test_01_malicious_prompts_no_mitigation(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return service.without_guidelines()
    
    results = run_prompt_analysis_test(
        test_id=1,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )