from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from tests.integration.test_utils import run_prompt_analysis_test


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

    results = run_prompt_analysis_test(
        test_id=3,
        model_id=ModelId.MICROSOFT_PHI_3_MINI4K_INSTRUCT,
        text_generation_completion_service=text_generation_completion_service,
        semantic_similarity_service=semantic_similarity_service,
        prompts=malicious_prompts,
        comparison_texts=prompt_injection_all_example_responses,
        service_configurator=configure_service
    )