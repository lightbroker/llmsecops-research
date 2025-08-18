from src.text_generation.common.model_id import ModelId
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from tests.integration.test_utils import get_prompt_batch, run_prompt_analysis_test


import pytest


@pytest.mark.mitigation
def test_01_malicious_prompts_no_mitigation_apple_openelm_3b_instruct(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return service.without_guidelines()

    results = run_prompt_analysis_test(
        test_id=1,
        model_id=ModelId.APPLE_OPENELM_3B_INSTRUCT,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=get_prompt_batch(malicious_prompts),
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )