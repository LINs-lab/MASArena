"""Tests for benchmark runner."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from mas_arena.benchmark_runner import BenchmarkRunner
from mas_arena.evaluators import BENCHMARKS


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""
    
    def test_benchmark_runner_init(self, temp_dir):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        assert runner.results_dir == str(temp_dir)
        assert runner.metrics_registry is not None
        assert runner.metrics_collector is not None
    
    def test_setup_metrics(self, temp_dir):
        """Test metrics setup."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        registry = runner._setup_metrics()
        assert registry is not None
    
    def test_prepare_benchmark_valid(self, temp_dir):
        """Test benchmark preparation with valid parameters."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Create a temporary test data file
        test_data = [{"id": "test_001", "problem": "What is 2+2?", "solution": "4"}]
        data_file = temp_dir / "test_data.jsonl"
        with open(data_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Test preparation
        agent, problems, benchmark_config, output_file = runner._prepare_benchmark(
            benchmark_name="math",
            data_path=str(data_file),
            limit=1,
            agent_system="single_agent",
            agent_config={},
            verbose=False
        )
        
        assert agent is not None
        assert len(problems) == 1
        assert problems[0]["id"] == "test_001"
        assert benchmark_config is not None
        assert output_file is not None
    
    def test_prepare_benchmark_invalid_benchmark(self, temp_dir):
        """Test benchmark preparation with invalid benchmark name."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        with pytest.raises(ValueError, match="Unknown benchmark"):
            runner._prepare_benchmark(
                benchmark_name="invalid_benchmark",
                data_path=None,
                limit=1,
                agent_system="single_agent",
                agent_config={},
                verbose=False
            )
    
    def test_prepare_benchmark_invalid_agent(self, temp_dir):
        """Test benchmark preparation with invalid agent system."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Create a temporary test data file
        test_data = [{"id": "test_001", "problem": "What is 2+2?", "solution": "4"}]
        data_file = temp_dir / "test_data.jsonl"
        with open(data_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        with pytest.raises(ValueError, match="Unknown agent system"):
            runner._prepare_benchmark(
                benchmark_name="math",
                data_path=str(data_file),
                limit=1,
                agent_system="invalid_agent",
                agent_config={},
                verbose=False
            )
    
    def test_prepare_benchmark_missing_data_file(self, temp_dir):
        """Test benchmark preparation with missing data file."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        with pytest.raises(FileNotFoundError):
            runner._prepare_benchmark(
                benchmark_name="math",
                data_path="nonexistent_file.jsonl",
                limit=1,
                agent_system="single_agent",
                agent_config={},
                verbose=False
            )
    
    @pytest.mark.asyncio
    async def test_process_one_problem(self, temp_dir, sample_problem):
        """Test processing a single problem."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.evaluate = AsyncMock(return_value={
            "extracted_answer": "4",
            "score": 1.0,
            "is_correct": True,
            "status": "success",
            "reasoning": "Test reasoning",
            "execution_time_ms": 100,
            "llm_usage": {}
        })
        
        benchmark_config = {
            "normalization_keys": {
                "problem": "problem",
                "solution": "solution",
                "id": "id"
            }
        }
        
        result = await runner._process_one_problem(
            i=0,
            p=sample_problem,
            agent=mock_agent,
            benchmark_config=benchmark_config,
            verbose=False
        )
        
        assert result["problem_id"] == "test_001"
        assert result["score"] == 1.0
        assert result["is_correct"] == True
        assert result["agent_system"] == "test_agent"
        mock_agent.evaluate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_one_problem_error(self, temp_dir, sample_problem):
        """Test processing a problem that raises an error."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Mock agent that raises an error
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.evaluate = AsyncMock(side_effect=Exception("Test error"))
        
        benchmark_config = {"normalization_keys": {}}
        
        result = await runner._process_one_problem(
            i=0,
            p=sample_problem,
            agent=mock_agent,
            benchmark_config=benchmark_config,
            verbose=False
        )
        
        assert result["problem_id"] == "problem_1"
        assert result["status"] == "error"
        assert result["score"] == 0
        assert "error" in result
    
    def test_finalize_benchmark(self, temp_dir):
        """Test benchmark finalization."""
        runner = BenchmarkRunner(results_dir=str(temp_dir))
        
        # Mock results
        all_results = [
            {"score": 1.0, "duration_ms": 100, "status": "success"},
            {"score": 0.0, "duration_ms": 150, "status": "success"},
            {"score": 1.0, "duration_ms": 120, "status": "success"},
        ]
        
        output_file = temp_dir / "test_results.json"
        
        summary = runner._finalize_benchmark(
            all_results=all_results,
            benchmark_name="test_benchmark",
            agent_system="test_agent",
            output_file=output_file,
            verbose=False
        )
        
        assert summary["total_problems"] == 3
        assert summary["correct"] == 2
        assert summary["accuracy"] == 2/3
        assert summary["total_duration_ms"] == 370
        assert summary["avg_duration_ms"] == 370/3
        
        # Check that results file was created
        assert output_file.exists()
        
        # Check file contents
        with open(output_file) as f:
            saved_data = json.load(f)
        assert "summary" in saved_data
        assert "results" in saved_data
        assert len(saved_data["results"]) == 3


class TestBenchmarkRunnerIntegration:
    """Test benchmark runner integration scenarios."""
    
    def test_custom_json_serializer(self):
        """Test custom JSON serializer handles various object types."""
        from mas_arena.benchmark_runner import custom_json_serializer
        from datetime import datetime
        from pathlib import Path
        
        # Test datetime
        dt = datetime.now()
        result = custom_json_serializer(dt)
        assert isinstance(result, str)
        
        # Test Path
        path = Path("/test/path")
        result = custom_json_serializer(path)
        assert isinstance(result, str)
        
        # Test object with __dict__
        class TestObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"
        
        obj = TestObj()
        result = custom_json_serializer(obj)
        assert isinstance(result, dict)
        assert result["attr1"] == "value1"
        assert result["attr2"] == "value2"