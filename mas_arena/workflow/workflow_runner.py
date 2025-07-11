# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/evaluator.py) under MIT License

import asyncio
import json

from tqdm.asyncio import tqdm_asyncio, tqdm
from typing import Tuple, Optional, Callable

from mas_arena.core.base_llm import BaseLLM
from mas_arena.core.benchmark import Benchmark
from mas_arena.core.llm_utils import cost_manager
from mas_arena.evaluators.base_evaluator import BaseEvaluator


class WorkflowRunner:

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    def _configure_graph(self, graph, evaluator):
        return graph(name=evaluator.name, llm_config=self.llm.config, evaluator=evaluator)

    async def graph_evaluate_async(self, evaluator: BaseEvaluator, graph: Callable, is_test: bool = False,
                                   max_concurrent_tasks: int = 20) -> Tuple[float, float, float]:

        configured_graph = self._configure_graph(graph=graph, evaluator=evaluator)

        # get data for evaluation
        from mas_arena.evaluators import BENCHMARKS
        benchmark_config = BENCHMARKS[evaluator.name]
        data_path = benchmark_config.get("data_path", f"data/{evaluator.name}_test.jsonl")
        data = []
        try:
            with open(data_path, "r") as f:
                data = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not data or len(data) == 0:
            print("No data to evaluate. Returning zeros.")
            return (0.0, 0.0, 0.0, True)

        cost_before = cost_manager.get_total_cost()

        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def evaluate_with_semaphore(example, i: int = None):
            async with semaphore:
                try:
                    return await evaluator.async_evaluate(configured_graph, example, i)
                except Exception as e:
                    print(f"Evaluation failed: {str(e)}")
                    return None

        # Create tasks for concurrent execution with semaphore

        tasks = []
        for i,example in enumerate(data):
            tasks.append(evaluate_with_semaphore(example,i))

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
