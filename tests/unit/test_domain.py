import pytest
from src.text_generation.domain.guardrail_processed_completion import GuardrailProcessedCompletion


@pytest.mark.unit
def test_guardrail_analyzed_response():
    response = GuardrailProcessedCompletion(
        score=0.72839,
        cosine_similarity_risk_threshold=0.5,
        original_completion="compromised response",
        final="I can't answer that"
    )
    assert response.is_original_completion_malicious == True