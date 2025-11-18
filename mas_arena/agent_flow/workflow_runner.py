# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/evaluator.py) under MIT License

import asyncio
import json

from tqdm.asyncio import tqdm_asyncio
from typing import Tuple, Optional, Callable, Any

from mas_arena.agents import AgentSystem
from mas_arena.evaluators.utils.normalization import normalize_problem_keys
from mas_arena.utils.llm_utils import cost_manager
from mas_arena.evaluators.base_evaluator import BaseEvaluator


class WorkflowRunner:

    def __init__(self, agent: Optional[AgentSystem] = None):
        self.agent = agent

    def _configure_graph(self, graph, evaluator):
        return graph(name=evaluator.name, agent_name="single_agent", evaluator=evaluator)

    async def graph_evaluate_async(self, evaluator: BaseEvaluator, graph: Callable, is_test: bool = False,
                                   max_concurrent_tasks: int = 20, train_size: int = 40, test_size: int = 20) -> Tuple[float, float, float]:
        configured_graph = self._configure_graph(graph=graph, evaluator=evaluator)

        data = evaluator.get_test_data(sample_size=test_size) if is_test else evaluator.get_dev_data(sample_size=train_size)
        if not data or len(data) == 0:
            print("No data to evaluate. Returning zeros.")
            return (0.0, 0.0, 0.0, True)

        cost_before = cost_manager.get_total_cost()

        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def evaluate_with_semaphore(problem, i: int = None):
            async with semaphore:
                try:
                    return await self.run_and_evaluate_problem(evaluator, configured_graph, problem, i)
                except Exception as e:
                    print(f"Evaluation failed: {str(e)}")
                    return None

        # Create tasks for concurrent execution with semaphore

        tasks = []
        for i,problem in enumerate(data):
            tasks.append(evaluate_with_semaphore(problem,i))

        # Wait for all tasks to complete
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Evaluating {evaluator.name} problems",
            total=len(data)
        )

        # Replace failed evaluations (None results) with 0
        valid_results = [0.0 if r is None else r for r in results]
        all_failed = all(r is None for r in results)

        # get total cost after evaluation
        total_cost = cost_manager.get_total_cost() - cost_before
        avg_cost = total_cost / len(data)

        if not valid_results:
            print("No valid results. Returning zeros.")
            avg_metrics = 0.0
        else:
            avg_metrics = sum(valid_results) / len(valid_results)

        return avg_metrics, avg_cost, total_cost, all_failed

    async def run_and_evaluate_problem(
            self,
            evaluator: BaseEvaluator,
            graph: Callable,
            problem: Any,
            i: int = None) -> float:
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        solution = await graph(prompt, entry_point)
        run_result = {"final_answer": solution, "extracted": True}
        from mas_arena.evaluators import BENCHMARKS
        benchmark_config = BENCHMARKS.get(evaluator.name, {})
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(problem, key_mapping, i)
        result = await evaluator.async_evaluate(normalized_problem, run_result)
        return result["score"] if "score" in result else 0.0
