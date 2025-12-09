import sys
import os
import asyncio
# 保证可以import mas_arena
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

if __name__ == "__main__":
    agent_config = {
        "num_agents": 3,
        "model_name": "gpt-4.1",
        "manager_tools": ["python_interpreter", "final_answer"],
        "search_tools": ["search", "wikipedia"],
        "memory": None,
        "verbosity_level": 1,
        "evaluator": "math",
    }
    summary = BenchmarkRunner().arun(
        benchmark_name="math",
        agent_system="swarm",
        agent_config=agent_config,
        limit=5,
        pass_at_k=1,
        log_file="logs/swarm_simple.log",
    )
    print("==== Swarm简易测试结果 ====")
    print(summary)