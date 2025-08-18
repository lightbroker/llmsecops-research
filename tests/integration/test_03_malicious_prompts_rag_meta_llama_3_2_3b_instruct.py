from src.text_generation.common.model_id import ModelId
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from tests.integration.test_utils import run_prompt_analysis_test


import pytest


@pytest.mark.mitigation
def test_03_malicious_prompts_rag_meta_llama_3_2_3b_instruct(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return (service
                .with_rag_context_guidelines())

    results = run_prompt_analysis_test(
        test_id=3,
        model_id=ModelId.META_LLAMA_3_2_3B_INSTRUCT,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )