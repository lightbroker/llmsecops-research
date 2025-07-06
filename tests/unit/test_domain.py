import pytest
from src.text_generation.domain.average import Average
from src.text_generation.domain.guardrail_analyzed_response import GuardrailAnalyzedResponse


@pytest.mark.unit
def test_get_average():
    scores = [
        0.12765,
        0.00282,
        0.63945,
        0.97123,
        0.38921
    ]
    avg_1 = Average().from_list_of_floats(scores)
    assert avg_1 == 0.426072


@pytest.mark.unit
def test_guardrail_analyzed_response():
    response = GuardrailAnalyzedResponse(
        score=0.72839,
        cosine_similarity_risk_threshold=0.5,
        original="compromised response",
        final="I can't answer that"
    )
    assert response.is_malicious == True