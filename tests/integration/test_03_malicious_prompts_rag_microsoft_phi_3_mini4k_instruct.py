import pytest
from src.text_generation.common.model_id import ModelId
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from tests.integration.test_utils import get_prompt_batch, run_prompt_analysis_test


@pytest.mark.mitigation
def test_03_malicious_prompts_rag_microsoft_phi_3_mini4k_instruct(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return (service
                .with_rag_context_guidelines())

    prompt_batch, start, end = get_prompt_batch(malicious_prompts)
    
    results = run_prompt_analysis_test(
        test_id=3,
        model_id=ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=prompt_batch,
        start=start,
        end=end,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )