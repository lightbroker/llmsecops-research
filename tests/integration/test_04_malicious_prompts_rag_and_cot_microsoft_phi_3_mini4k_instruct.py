import os
import pytest

from src.text_generation.common.model_id import ModelId
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from tests.integration.test_utils import run_prompt_analysis_test, get_prompt_batch


@pytest.mark.mitigation
def test_04_malicious_prompts_rag_and_cot_microsoft_phi_3_mini4k_instruct(
    text_generation_completion_service: AbstractTextGenerationCompletionService,
    semantic_similarity_service,
    malicious_prompts,
    prompt_injection_all_example_responses
):
    def configure_service(service):
        return (service
            .with_chain_of_thought_guidelines()
            .with_rag_context_guidelines())
    
    results = run_prompt_analysis_test(
        test_id=4,
        model_id=ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=get_prompt_batch(malicious_prompts),
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )