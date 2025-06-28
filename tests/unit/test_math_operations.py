import pytest
from src.text_generation.domain.average import Average


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
