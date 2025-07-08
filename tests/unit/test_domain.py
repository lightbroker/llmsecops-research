import pytest
from src.text_generation.domain.guardrail_analyzed_response import GuardrailAnalyzedResponse


@pytest.mark.unit
def test_guardrail_analyzed_response():
    response = GuardrailAnalyzedResponse(
        score=0.72839,
        cosine_similarity_risk_threshold=0.5,
        original="compromised response",
        final="I can't answer that"
    )
    assert response.is_completion_malicious == True