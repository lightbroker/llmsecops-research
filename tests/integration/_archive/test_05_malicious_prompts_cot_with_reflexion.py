"""
    Usage:
    $ pytest -k test_05_malicious_prompts_cot_with_reflexion -s --disable-warnings
"""

import pytest
from test_utils import run_prompt_analysis_test
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


@pytest.mark.mitigation
def test_05_malicious_prompts_cot_with_reflexion(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service: AbstractTextGenerationCompletionService):
        return (service
                .with_chain_of_thought_guidelines()
                .with_reflexion_guardrails())
    
    results = run_prompt_analysis_test(
        test_id=5,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )

