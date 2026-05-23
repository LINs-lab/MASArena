import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mas_arena.benchmark_runner import BenchmarkRunner
from mas_arena.agents.evoagent_newcore import EvoAgent
from mas_arena.tools.browser_tool import BrowserTool


@pytest.mark.asyncio
async def test_process_one_problem_timeout_scores_zero(temp_dir, sample_problem):
    runner = BenchmarkRunner(results_dir=str(temp_dir), problem_timeout_seconds=0.01)

    mock_agent = Mock()
    mock_agent.name = "test_agent"

    async def slow_evaluate(*args, **kwargs):
        await asyncio.sleep(1)

    mock_agent.evaluate = AsyncMock(side_effect=slow_evaluate)

    result = await runner._process_one_problem(
        i=0,
        p=sample_problem,
        agent=mock_agent,
        benchmark_config={"normalization_keys": {}},
        verbose=False,
    )

    assert result["problem_id"] == "problem_1"
    assert result["status"] == "timeout"
    assert result["score"] == 0
    assert result["is_correct"] is False


@pytest.mark.asyncio
async def test_browser_tool_action_timeout_returns_error(monkeypatch):
    monkeypatch.setenv("MAS_ARENA_BROWSER_TIMEOUT_SECONDS", "0.01")

    async def slow_get_current_url():
        await asyncio.sleep(1)
        return "https://example.com"

    tool = SimpleNamespace(
        available=True,
        browser_instance=SimpleNamespace(get_current_url=slow_get_current_url),
    )

    result = await BrowserTool.forward(tool, action="get_url", url=None)

    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_evoagent_worker_timeout_is_configurable():
    evo_agent = EvoAgent.__new__(EvoAgent)
    evo_agent.agent_task_timeout_seconds = 0.01
    evo_agent.evaluator = Mock()
    evo_agent.evaluator.calculate_score.return_value = (0.0, "")

    async def slow_solve(_problem):
        await asyncio.sleep(0.05)
        return {"extracted_answer": "late answer"}

    worker = SimpleNamespace(
        agent_id="worker-1",
        name="EVO-test",
        score=0.0,
        result={},
        solve=slow_solve,
    )

    await EvoAgent._run_agent_task(evo_agent, worker, "question", {"solution": "answer"})

    assert worker.score == 0.0
    assert worker.result["status"] == "timeout"
    assert worker.result["extracted_answer"] == "Execution timeout, unable to get answer"


@pytest.mark.asyncio
async def test_evoagent_staggers_worker_start_times():
    evo_agent = EvoAgent.__new__(EvoAgent)
    evo_agent.worker_delay_seconds = 0.01

    async def marker(value):
        return value

    start = time.perf_counter()
    results = await asyncio.gather(*EvoAgent._stagger(evo_agent, [marker(1), marker(2)]))

    assert results == [1, 2]
    assert time.perf_counter() - start >= 0.009
