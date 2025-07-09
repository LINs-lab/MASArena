# Contributing to MASArena

We welcome contributions to MASArena! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LINs-lab/MASArena.git
   cd MASArena
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   source .venv/bin/activate
   
   # Or using pip
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Install development dependencies:**
   ```bash
   pip install pytest pytest-cov pytest-asyncio ruff black isort
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Testing

### Running Tests

We use pytest for testing. Here are the most common test commands:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mas_arena --cov-report=html

# Run only unit tests (fast)
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Run tests excluding slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_agents.py

# Run with verbose output
pytest -v

# Run tests in parallel (if pytest-xdist is installed)
pytest -n auto
```

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast, isolated tests for individual components
- **Integration Tests** (`@pytest.mark.integration`): Tests for component interactions
- **Slow Tests** (`@pytest.mark.slow`): Long-running tests that may involve external services

### Writing Tests

#### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_agents.py           # Tests for agent systems
├── test_evaluators.py       # Tests for evaluation components
├── test_tools.py            # Tests for tool management
├── test_benchmark_runner.py # Tests for benchmark execution
└── test_integration.py      # End-to-end integration tests
```

#### Test Guidelines

1. **Use descriptive test names:**
   ```python
   def test_agent_initialization_with_valid_config():
       # Test implementation
   ```

2. **Use fixtures for common setup:**
   ```python
   def test_agent_run(mock_agent_config, disable_llm_calls):
       agent = SingleAgent(config=mock_agent_config)
       # Test implementation
   ```

3. **Mock external dependencies:**
   ```python
   @patch('mas_arena.agents.base.ChatOpenAI')
   def test_agent_with_mocked_llm(mock_llm):
       # Test implementation
   ```

4. **Mark tests appropriately:**
   ```python
   @pytest.mark.unit
   def test_fast_unit_test():
       pass
   
   @pytest.mark.integration
   def test_component_integration():
       pass
   
   @pytest.mark.slow
   def test_long_running_operation():
       pass
   
   @pytest.mark.asyncio
   async def test_async_function():
       pass
   ```

#### Available Fixtures

See `tests/conftest.py` for available fixtures:

- `temp_dir`: Temporary directory for test files
- `sample_problem`: Sample problem data
- `sample_math_problems`: Multiple math problems
- `mock_llm_response`: Mock LLM response
- `mock_agent_config`: Mock agent configuration
- `disable_llm_calls`: Disable actual LLM API calls
- `disable_file_operations`: Disable file operations

## Code Quality

### Linting and Formatting

We use several tools to maintain code quality:

```bash
# Run all linting tools
ruff check .          # Linting
black .               # Code formatting
isort .               # Import sorting

# Fix issues automatically
ruff check . --fix
black .
isort .
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Contribution Types

### 🧠 Adding New Agent Systems

1. Create a new file in `mas_arena/agents/`
2. Inherit from `AgentSystem` or `SingleAgent`
3. Implement required methods
4. Add tests in `tests/test_agents.py`
5. Update documentation

### 📊 Adding New Benchmarks

1. Add data files to `data/`
2. Create evaluator in `mas_arena/evaluators/`
3. Update benchmark configuration
4. Add tests in `tests/test_evaluators.py`

### 🛠 Adding New Tools

1. Create tool in `mas_arena/tools/`
2. Inherit from `BaseTool`
3. Implement `execute()` and `get_schema()` methods
4. Add tests in `tests/test_tools.py`
5. Register tool in tool manager

### 🐛 Bug Fixes

1. Write a test that reproduces the bug
2. Fix the bug
3. Ensure the test passes
4. Run the full test suite

## Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the guidelines above
3. **Write or update tests** for your changes
4. **Run the test suite** and ensure all tests pass:
   ```bash
   pytest
   ruff check .
   black --check .
   isort --check-only .
   ```
5. **Update documentation** if necessary
6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots for UI changes (if applicable)

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated (if needed)
- [ ] Changelog updated (for significant changes)
- [ ] No merge conflicts

## Continuous Integration

Our CI pipeline runs:

1. **Tests** on Python 3.11 and 3.12
2. **Linting** with ruff, black, and isort
3. **Coverage** reporting to Codecov
4. **Documentation** building and deployment

All checks must pass before merging.

## Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check our [documentation](https://lins-lab.github.io/MASArena)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building this together! 🚀

## License

By contributing, you agree that your contributions will be licensed under the MIT License.