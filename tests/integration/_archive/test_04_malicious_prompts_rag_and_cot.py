"""
    Usage:
    $ pytest -k test_04_malicious_prompts_rag_and_cot -s --disable-warnings
"""

import pytest
from src.text_generation.common.model_id import ModelId
from test_utils import run_prompt_analysis_test
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService


@pytest.mark.mitigation
@pytest.mark.parametrize("model_id", [
    ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT,
    ModelId.APPLE_OPENELM_1_1B_INSTRUCT,
    ModelId.META_LLAMA_3_2_3B_INSTRUCT
])
def test_04_malicious_prompts_rag_and_cot(
    model_id: ModelId,
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
        model_id=model_id,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts[:1],
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )