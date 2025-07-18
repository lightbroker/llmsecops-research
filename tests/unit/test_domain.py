import pytest
from src.text_generation.domain.guardrails_result import GuardrailsResult


@pytest.mark.unit
def test_guardrail_analyzed_response():
    response = GuardrailsResult(
        cosine_similarity_score=0.72839,
        cosine_similarity_risk_threshold=0.5,
        original_completion="compromised response",
        guardrails_processed_completion_text="I can't answer that"
    )
    assert response.is_original_completion_malicious == True