# Test Workflow in GitHub Actions


## Table of Contents

- [Testing Framework](#testing-framework)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Running Tests Locally](#running-tests-locally)
- [Code Quality Standards](#code-quality-standards)
- [Contributing Workflow](#contributing-workflow)


## Testing Framework

### Framework Choice

We use **pytest** as our primary testing framework because:
- Excellent support for async testing with `pytest-asyncio`
- Rich fixture system for test setup and teardown
- Comprehensive assertion introspection
- Extensive plugin ecosystem
- Built-in coverage reporting

### Test Structure

Our test suite is organized into the following categories:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_agents.py           # Agent system tests
├── test_evaluators.py       # Evaluator and benchmark tests
└── test_benchmark_runner.py # Benchmark runner integration tests
```

### Test Categories

1. **Unit Tests** (`@pytest.mark.unit`)
   - Test individual functions and classes in isolation
   - Fast execution, no external dependencies
   - Mock external services and APIs

2. **Integration Tests** (`@pytest.mark.integration`)
   - Test component interactions
   - May involve file I/O or network calls
   - Use test fixtures and temporary directories

3. **Async Tests** (`@pytest.mark.asyncio`)
   - Test asynchronous functionality
   - Agent evaluation and benchmark processing
   - Concurrent execution scenarios

### Key Test Cases

Our test suite covers:

- **Agent Systems**: Creation, configuration, and evaluation interfaces
- **Evaluators**: Math, GSM8K, and other benchmark evaluators
- **Benchmark Runner**: Problem processing, result aggregation, and error handling
- **Registry Systems**: Agent and benchmark registration
- **Utilities**: JSON serialization, file handling, and metrics collection

## GitHub Actions CI/CD

### Workflow Overview

Our CI/CD pipeline (`.github/workflows/test.yml`) automatically runs on:
- **Push** to `main` and `develop` branches
- **Pull requests** targeting `main` and `develop` branches

### Workflow Jobs

#### 1. Test Job

**Matrix Strategy**: Tests run on Python 3.11 and 3.12

**Steps**:
1. **Checkout**: Get the latest code
2. **Python Setup**: Install specified Python version
3. **Dependency Caching**: Cache pip dependencies for faster builds
4. **Install Dependencies**: Install project and test dependencies
5. **Environment Setup**: Set `PYTHONPATH` and test mode flags
6. **Unit Tests**: Run individual test files with verbose output
7. **Coverage Tests**: Generate coverage reports
8. **Upload Coverage**: Send coverage data to Codecov (Python 3.11 only)

#### 2. Lint Job

**Purpose**: Ensure code quality and formatting consistency

**Steps**:
1. **Ruff Linting**: Check code style and potential issues
2. **Format Checking**: Verify code formatting (non-blocking)

### Caching Strategy

We implement dependency caching to improve build performance:
- **Cache Key**: Based on `requirements.txt` hash
- **Cache Path**: `~/.cache/pip`
- **Fallback**: OS-specific pip cache

### Environment Variables

- `PYTHONPATH`: Ensures proper module imports
- `MAS_ARENA_TEST_MODE`: Enables test-specific configurations
- API keys are mocked in test environment

## Running Tests Locally

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py -v

# Run tests with coverage
pytest --cov=mas_arena --cov-report=html

# Run only unit tests
pytest -m unit

# Run async tests
pytest -m asyncio
```

### Test Configuration

Our `pytest.ini` configuration:
- **Test Discovery**: `test_*.py` files, `Test*` classes, `test_*` functions
- **Markers**: `slow`, `integration`, `unit`, `asyncio`
- **Warnings**: Filtered deprecation warnings
- **Options**: Strict marker enforcement, quiet output

### Debugging Tests

```bash
# Verbose output with full traceback
pytest -v --tb=long

# Stop on first failure
pytest -x

# Run specific test method
pytest tests/test_agents.py::TestAgentCreation::test_create_single_agent
```

## Code Quality Standards

### Linting with Ruff

We use **Ruff** for fast Python linting and formatting:

```bash
# Check code style
ruff check mas_arena/

# Format code
ruff format mas_arena/

# Check formatting without changes
ruff format --check mas_arena/
```

### Configuration

- **Line Length**: 120 characters
- **Target Version**: Python 3.11+
- **Rules**: Standard Python style guidelines

### Coverage Requirements

- Maintain test coverage above 80%
- New features must include corresponding tests
- Critical paths require comprehensive test coverage

## Contributing Workflow

### Before Submitting a Pull Request

1. **Run Tests Locally**:
   ```bash
   pytest tests/ -v
   ```

2. **Check Code Quality**:
   ```bash
   ruff check mas_arena/
   ruff format mas_arena/
   ```

3. **Verify Coverage**:
   ```bash
   pytest --cov=mas_arena --cov-report=term-missing
   ```

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Make Changes**: Implement your feature with tests
3. **Test Locally**: Ensure all tests pass
4. **Commit Changes**: Use descriptive commit messages
5. **Push Branch**: `git push origin feature/your-feature-name`
6. **Create PR**: Submit pull request with description
7. **CI Validation**: Wait for GitHub Actions to pass
8. **Code Review**: Address reviewer feedback
9. **Merge**: Squash and merge after approval

### Writing New Tests

When adding new functionality:

1. **Create Test File**: Follow naming convention `test_*.py`
2. **Use Fixtures**: Leverage shared fixtures from `conftest.py`
3. **Mock External Dependencies**: Use `unittest.mock` for API calls
4. **Test Edge Cases**: Include error conditions and boundary cases
5. **Add Markers**: Use appropriate pytest markers
6. **Document Tests**: Include docstrings for complex test scenarios

### Example Test Structure

```python
"""Tests for new feature."""

import pytest
from unittest.mock import Mock, patch

from mas_arena.your_module import YourClass


class TestYourClass:
    """Test your new class functionality."""
    
    def test_basic_functionality(self, sample_config):
        """Test basic functionality with valid input."""
        instance = YourClass(sample_config)
        result = instance.method()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_method(self, sample_config):
        """Test async method execution."""
        instance = YourClass(sample_config)
        result = await instance.async_method()
        assert result["status"] == "success"
    
    def test_error_handling(self, sample_config):
        """Test error handling for invalid input."""
        instance = YourClass(sample_config)
        with pytest.raises(ValueError):
            instance.method(invalid_input=True)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **API Key Errors**: Check that test environment uses mocked APIs
3. **Async Test Failures**: Verify `pytest-asyncio` is installed
4. **Coverage Issues**: Exclude test files from coverage reports

### Getting Help

- Check existing issues and discussions
- Review test output and error messages
- Consult project documentation
- Ask questions in pull request comments
