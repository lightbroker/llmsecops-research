"""
pytest -v tests/integration/test_domain.py
"""

import pytest
from unittest.mock import MagicMock

from src.text_generation.domain.guardrails_result import GuardrailsResult
from src.text_generation.domain.guidelines_result import GuidelinesResult
from src.text_generation.domain.original_completion_result import OriginalCompletionResult
from src.text_generation.domain.text_generation_completion_result import TextGenerationCompletionResult


class TestTextGenerationCompletionResult:
    """Test suite for TextGenerationCompletionResult and related classes."""
    
    @pytest.fixture
    def sample_llm_config(self):
        """Sample LLM configuration for testing."""
        return {
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100
        }
    
    @pytest.fixture
    def sample_full_prompt(self):
        """Sample full prompt for testing."""
        return {
            "system": "You are a helpful assistant",
            "user": "Test prompt"
        }
    
    def test_original_result_only(self, sample_llm_config, sample_full_prompt):
        """Test TextGenerationCompletionResult with only original result filled in."""
        # Arrange
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.5
        original_result.cosine_similarity_risk_threshold=0.7

        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=None,
            guardrails_result=None
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == "Original completion text"
        assert original_result.user_prompt == "Test user prompt"
        assert original_result.llm_config == sample_llm_config
        assert not original_result.is_completion_malicious()  # 0.5 < 0.7
    
    def test_guidelines_and_original_guidelines_precedence(self, sample_llm_config, sample_full_prompt):
        """Test that guidelines result takes precedence over original when both are present."""
        # Arrange
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.8
        original_result.cosine_similarity_risk_threshold=0.7

        guidelines_result = GuidelinesResult(
            user_prompt="Test user prompt",
            completion_text="Guidelines processed completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )

        guidelines_result.cosine_similarity_score=0.6
        guidelines_result.cosine_similarity_risk_threshold=0.7
        
        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=guidelines_result,
            guardrails_result=None
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == "Guidelines processed completion text"
        assert original_result.is_completion_malicious()  # 0.8 >= 0.7
        assert not guidelines_result.is_completion_malicious()  # 0.6 < 0.7
    
    def test_guardrails_guidelines_original_guardrails_precedence(self, sample_llm_config, sample_full_prompt):
        """Test that guardrails result takes precedence when all three are present."""
        # Arrange
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.9
        original_result.cosine_similarity_risk_threshold=0.7
        
        guidelines_result = GuidelinesResult(
            user_prompt="Test user prompt",
            completion_text="Guidelines processed completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )

        guidelines_result.cosine_similarity_score=0.8
        guidelines_result.cosine_similarity_risk_threshold=0.7
        
        guardrails_result = GuardrailsResult(
            user_prompt="Test user prompt",
            completion_text="Guardrails processed completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=guidelines_result,
            guardrails_result=guardrails_result
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == "Guardrails processed completion text"
        assert original_result.is_completion_malicious()  # 0.9 >= 0.7
        assert guidelines_result.is_completion_malicious()  # 0.8 >= 0.7
    
    def test_empty_completion_fallback_behavior(self, sample_llm_config, sample_full_prompt):
        """Test fallback behavior when some completion texts are empty."""
        # Arrange - guardrails has empty text, should fall back to guidelines
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.5
        original_result.cosine_similarity_risk_threshold=0.7
        
        guidelines_result = GuidelinesResult(
            user_prompt="Test user prompt",
            completion_text="Guidelines processed completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )

        guidelines_result.cosine_similarity_score=0.6
        guidelines_result.cosine_similarity_risk_threshold=0.7
        
        guardrails_result = GuardrailsResult(
            user_prompt="Test user prompt",
            completion_text="",  # Empty string
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=guidelines_result,
            guardrails_result=guardrails_result
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == "Guidelines processed completion text"
    
    def test_whitespace_only_completion_fallback(self, sample_llm_config, sample_full_prompt):
        """Test fallback behavior when completion text is only whitespace."""
        # Arrange
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.5
        original_result.cosine_similarity_risk_threshold=0.7
        
        guidelines_result = GuidelinesResult(
            user_prompt="Test user prompt",
            completion_text="   \n\t  ",  # Whitespace only
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )

        guidelines_result.cosine_similarity_score=0.6
        guidelines_result.cosine_similarity_risk_threshold=0.7
        
        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=guidelines_result,
            guardrails_result=None
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == "Original completion text"
    
    def test_all_empty_completions_returns_empty_string(self, sample_llm_config, sample_full_prompt):
        """Test that empty string is returned when all completions are empty."""
        # Arrange
        original_result = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result.cosine_similarity_score=0.5
        original_result.cosine_similarity_risk_threshold=0.7
        
        completion_result = TextGenerationCompletionResult(
            original_result=original_result,
            guidelines_result=None,
            guardrails_result=None
        )
        
        # Act
        completion_result.finalize_completion_text()
        final_text = completion_result.final_completion_text
        
        # Assert
        assert final_text == ""
    
    def test_risk_threshold_boundary_conditions(self, sample_llm_config, sample_full_prompt):
        """Test boundary conditions for risk threshold (exactly 0.7)."""
        # Test score exactly at threshold
        original_result_at_threshold = OriginalCompletionResult(
            user_prompt="Test user prompt",
            completion_text="Original completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config
        )
        
        original_result_at_threshold.cosine_similarity_score=0.7
        original_result_at_threshold.cosine_similarity_risk_threshold=0.7
        
        # Test score just below threshold
        guidelines_result_below_threshold = GuidelinesResult(
            user_prompt="Test user prompt",
            completion_text="Guidelines completion text",
            full_prompt=sample_full_prompt,
            llm_config=sample_llm_config,
            cosine_similarity_score=0.699999,  # Just below threshold
            cosine_similarity_risk_threshold=0.7
        )
        
        # Assert
        assert original_result_at_threshold.is_completion_malicious()  # 0.7 >= 0.7
        assert not guidelines_result_below_threshold.is_completion_malicious()  # 0.699999 < 0.7