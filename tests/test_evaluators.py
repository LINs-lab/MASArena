"""Tests for evaluator systems."""

import pytest
from unittest.mock import Mock, patch

from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.math_evaluator import MathEvaluator
from mas_arena.evaluators.gsm8k_evaluator import GSM8KEvaluator


class TestBenchmarkRegistry:
    """Test benchmark registry functionality."""
    
    def test_benchmarks_populated(self):
        """Test that BENCHMARKS dictionary is populated."""
        assert isinstance(BENCHMARKS, dict)
        assert len(BENCHMARKS) > 0
    
    def test_math_benchmark_exists(self):
        """Test that math benchmark is registered."""
        assert "math" in BENCHMARKS
        assert "evaluator" in BENCHMARKS["math"]
        assert "normalization_keys" in BENCHMARKS["math"]
    
    def test_gsm8k_benchmark_exists(self):
        """Test that GSM8K benchmark is registered."""
        assert "gsm8k" in BENCHMARKS
        assert "evaluator" in BENCHMARKS["gsm8k"]
        assert "normalization_keys" in BENCHMARKS["gsm8k"]


class TestBaseEvaluator:
    """Test base evaluator functionality."""
    
    def test_base_evaluator_init(self, sample_evaluator_config):
        """Test base evaluator initialization."""
        evaluator = BaseEvaluator("test_evaluator", sample_evaluator_config)
        assert evaluator.name == "test_evaluator"
        assert hasattr(evaluator, "evaluate")
    
    def test_base_evaluator_with_config(self):
        """Test base evaluator with configuration."""
        config = {"timeout": 30, "verbose": True}
        evaluator = BaseEvaluator("test_evaluator", config)
        assert evaluator.config == config


class TestMathEvaluator:
    """Test MathEvaluator class."""
    
    def test_math_evaluator_init(self, sample_evaluator_config):
        """Test MathEvaluator initialization."""
        evaluator = MathEvaluator("math", sample_evaluator_config)
        assert evaluator.name == "math"
        assert hasattr(evaluator, "math_equal")
    
    def test_math_answer_verification(self, sample_evaluator_config):
        """Test math answer verification."""
        evaluator = MathEvaluator("math", sample_evaluator_config)
        
        # Test correct answer
        assert evaluator.math_equal("4", "4") == True
        assert evaluator.math_equal("2+2", "4") == True
        
        # Test incorrect answer
        assert evaluator.math_equal("5", "4") == False
    
    def test_extract_answer(self, sample_evaluator_config):
        """Test answer extraction."""
        evaluator = MathEvaluator("math", sample_evaluator_config)
        
        # Test answer extraction
        text = "The answer is 42."
        answer = evaluator.extract_answer(text)
        assert answer is not None


class TestGSM8KEvaluator:
    """Test GSM8K evaluator functionality."""
    
    def test_gsm8k_evaluator_init(self, sample_evaluator_config):
        """Test GSM8K evaluator initialization."""
        evaluator = GSM8KEvaluator("gsm8k", sample_evaluator_config)
        assert evaluator.name == "gsm8k"
        assert hasattr(evaluator, "extract_number")
    
    def test_gsm8k_answer_extraction(self, sample_evaluator_config):
        """Test GSM8K answer extraction."""
        evaluator = GSM8KEvaluator("gsm8k", sample_evaluator_config)
        
        # Test various answer formats
        test_cases = [
            ("The answer is 42", 42.0),
            ("42.5", 42.5),
            ("The result is $123.45", 123.45),
            ("No number here", None)
        ]
        
        for text, expected in test_cases:
            result = evaluator.extract_number(text)
            assert result == expected


class TestEvaluatorIntegration:
    """Test evaluator integration functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluator_evaluate_method(self, sample_problem, sample_evaluator_config):
        """Test that evaluators have evaluate method."""
        evaluator = MathEvaluator("math", sample_evaluator_config)
        assert hasattr(evaluator, "evaluate")
        
        # Mock the evaluation to avoid complex dependencies
        with patch.object(evaluator, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = {
                "score": 1.0,
                "is_correct": True,
                "extracted_answer": "4",
                "message": "Correct answer"
            }
            
            run_result = {"final_answer": "4"}
            result = evaluator.evaluate(sample_problem, run_result)
            assert "score" in result
            assert "is_correct" in result