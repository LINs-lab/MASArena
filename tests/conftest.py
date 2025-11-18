"""Shared fixtures and configuration for tests."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["TAVILY_API_KEY"] = "test-key"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_problem():
    """Sample problem for testing evaluators."""
    return {
        "id": "test_001",
        "problem": "What is 2 + 2?",
        "solution": "4",
        "type": "math"
    }


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing."""
    return {
        "evaluator": "math",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30
    }


@pytest.fixture
def sample_evaluator_config():
    """Sample evaluator configuration for testing."""
    return {
        "data_path": "test_data.jsonl",
        "log_path": "test_logs",
        "timeout": 30
    }