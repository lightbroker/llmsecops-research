import pytest
from src.text_generation.domain.guardrails_processed_completion import GuardrailsProcessedCompletion


@pytest.mark.unit
def test_guardrail_analyzed_response():
    response = GuardrailsProcessedCompletion(
        score=0.72839,
        cosine_similarity_risk_threshold=0.5,
        original_completion="compromised response",
        final="I can't answer that"
    )
    assert response.is_original_completion_malicious == True