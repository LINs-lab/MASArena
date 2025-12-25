#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Runner

This module provides functionality for running benchmarks on agent systems.
"""

import os
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from openai.types.completion_usage import CompletionUsage
import traceback
from rich import print as rprint
import time
import logging

from mas_arena.metrics import MetricsRegistry, MetricsCollector
from mas_arena.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.utils.normalization import normalize_problem_keys


def custom_json_serializer(obj):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if hasattr(obj, "__dict__"):
        # For AIMessage-like objects, convert to dict
        return {key: getattr(obj, key) for key in obj.__dict__ if not key.startswith("_")}
    if isinstance(obj, CompletionUsage):
        return {
            "prompt_tokens": obj.prompt_tokens,
            "completion_tokens": obj.completion_tokens,
            "total_tokens": obj.total_tokens,
        }
    try:
        return str(obj)  # Fallback for other non-serializable types
    except Exception:
        return f"Object of type {type(obj).__name__} is not JSON serializable"


class BenchmarkRunner:
    """
    Simple interface for running benchmarks on multi-agent systems.

    Examples:
        >>> benchmark = BenchmarkRunner()
        >>> results = benchmark.run("math", limit=5)
    """

    def __init__(self, results_dir="results"):
        """
        Initialize the benchmark runner.

        Args:
            results_dir: Directory to save results
            metrics_dir: Directory to save metrics data
        """
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.agent_config = None  # Store agent configuration
        self.metrics_registry = None
        self.metrics_collector = None
        self._logging_setup = False

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        # os.makedirs(metrics_dir, exist_ok=True)

        # Set up metrics
        self.metrics_registry = self._setup_metrics()

        # Create centralized metrics collector
        self.metrics_collector = MetricsCollector(self.metrics_registry)

    def _setup_logging(self, log_file: str):
        """Set up logging to a file only (no console), and redirect stdout/stderr to the file."""
        if not log_file or self._logging_setup:
            return

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Configure root logger to write only to file
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Remove existing StreamHandlers (console) to avoid duplicate console output
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler):
                logger.removeHandler(h)

        # Add file handler if not already present for this file
        file_path_str = str(Path(log_file))
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == file_path_str for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Redirect stdout and stderr to the log file (only file)
        try:
            import sys as _sys
            file_stream = open(log_file, 'a', encoding='utf-8')
            _sys.stdout = file_stream
            _sys.stderr = file_stream
        except Exception:
            # If redirecting fails, ignore (logging to file still works)
            pass

        self._logging_setup = True

    def _setup_metrics(self):
        """Set up metrics collection"""
        registry = MetricsRegistry()
        return registry

    def _prepare_benchmark(
        self,
        benchmark_name,
        data_path,
        limit,
        agent_system,
        agent_config,
        verbose,
        data_id=None,
        memory_type=None,
    ):
        """
        Run a benchmark with the specified configuration.

        Args:
            benchmark_name: Name of the benchmark (math, drop, gsm8k, hotpotqa, humaneval, mbpp)
            data_path: Custom path to benchmark data file (optional)
            limit: Maximum number of problems to process
            agent_system: Agent system to use (single_agent, supervisor_mas, swarm)
            agent_config: Configuration for the agent system
            verbose: Whether to print progress information
            memory_type: Memory type to use (e.g., melo_memory)

        Returns:
            Dictionary of benchmark results
        """
        # Validate benchmark name
        if benchmark_name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Supported: {', '.join(BENCHMARKS.keys())}")

        benchmark_config = BENCHMARKS[benchmark_name]

        if verbose:
            print(f"Available agent systems: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")
        self.agent_config = agent_config or {}

        # Ensure the agent knows which evaluator to use by setting it in the config
        self.agent_config["evaluator"] = benchmark_name

        if not data_path:
            data_path = benchmark_config.get("data_path", f"data/{benchmark_name}_test.jsonl")

        output_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}.json"

        # Auto-download dataset if missing
        if not os.path.exists(data_path):
            if verbose:
                print(f"Data file not found at {data_path}. Attempting to download...")
            try:
                import importlib.util
                # Assume running from project root, check relative path first
                download_script_path = Path("data/download/download_dataset.py")
                if not download_script_path.exists():
                    # Fallback to finding it relative to this file
                    download_script_path = Path(__file__).parent.parent.parent / "data" / "download" / "download_dataset.py"
                
                if download_script_path.exists():
                    spec = importlib.util.spec_from_file_location("download_dataset", str(download_script_path))
                    if spec and spec.loader:
                        download_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(download_module)
                        if hasattr(download_module, "ensure_dataset_exists"):
                            download_module.ensure_dataset_exists(benchmark_name, str(data_path))
            except Exception as e:
                print(f"Warning: Failed to attempt automatic download: {e}")

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                problems = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_id:
            primary_id = benchmark_config.get("normalization_keys", {}).get("id", None)
            if primary_id is not None:
                for problem in problems:
                    if str(problem[primary_id]) == data_id:
                        problems = [problem]
                        break

        if limit and limit < len(problems):
            seed = 42
            random.seed(seed)
            problems = random.sample(problems, limit)

        return problems, benchmark_config, output_file

    async def _process_one_problem(self, i, p, agent, benchmark_config, verbose=True):
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(p, key_mapping, i)
        problem_id = normalized_problem["id"]

        if verbose:
            print(f"Problem {i + 1}: {problem_id}")

        self.metrics_collector.start_timer(f"mas_arena.problem.{problem_id}", {"problem_id": problem_id})

        try:
            results = await agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)
            results = results.raw_responses
            problem_duration_ms = self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")

            duration_ms = results.get("execution_time_ms", problem_duration_ms)
            score = results.get("score", 0)
            is_correct = results.get("is_correct", score == 1)

            self.metrics_collector.record_metric(
                "mas_arena.problem.result",
                score,
                {
                    "problem_id": problem_id,
                    "correct": is_correct,
                    "duration_ms": duration_ms,
                },
            )

            result_entry = {
                "problem_id": problem_id,
                "problem": normalized_problem["problem"],
                "expected": normalized_problem["solution"],
                "prediction": results.get("extracted_answer") or results.get("final_answer") or "",
                "score": score,
                "is_correct": is_correct,
                "status": results.get("status"),
                "reasoning": results.get("reasoning", ""),
                "duration_ms": duration_ms,
                "agent_system": agent.name,
                "llm_usage": results.get("llm_usage", {}),
                "summary": {
                    "correct": is_correct,
                    "score": score,
                    "duration_ms": duration_ms,
                },
            }
            if verbose:
                status_char = "E" if results.get("status") == "error" else "✓" if is_correct else "✗"
                print(f"Result: {status_char} ({duration_ms:.0f}ms)")
            return result_entry
        except Exception as e:
            self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")
            self.metrics_collector.record_error(
                "problem_processing",
                str(e),
                {"problem_id": problem_id, "error_type": type(e).__name__},
            )
            if verbose:
                print(f"Error processing problem {problem_id}: {e}")
                traceback.print_exc()
            return {
                "problem_id": problem_id,
                "problem": normalized_problem.get("problem"),
                "status": "error",
                "error": str(e),
                "score": 0,
            }

    async def _process_one_problem_pass_at_k(
        self,
        i,
        p,
        agent_system,
        agent_config,
        benchmark_config,
        verbose,
        k,
        memory_type=None,
    ):
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(p, key_mapping, i)
        problem_id = normalized_problem["id"]

        if verbose:
            print(f"Problem {i + 1}: {problem_id} (pass@{k})")

        self.metrics_collector.start_timer(f"mas_arena.problem.{problem_id}", {"problem_id": problem_id})

        try:
            async def run_one_sample():
                # Create a fresh agent instance per problem to isolate state
                agent = create_agent_system(agent_system, agent_config, memory_type=memory_type)
                agent.set_metrics_registry(self.metrics_registry)
                return await agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)

            evaluation_tasks = [run_one_sample() for _ in range(k)]
            results_list = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            problem_duration_ms = self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")

            valid_results = [res for res in results_list if not isinstance(res, Exception)]

            is_correct = any(res.get("is_correct", res.get("score") == 1) for res in valid_results)

            first_correct_result = next(
                (res for res in valid_results if res.get("is_correct", res.get("score") == 1)),
                None,
            )

            if first_correct_result:
                representative_result = first_correct_result
            else:
                representative_result = valid_results[0] if valid_results else {}

            total_duration = sum(res.get("execution_time_ms", 0) for res in valid_results)
            if not valid_results:
                total_duration = problem_duration_ms

            total_llm_usage = {}
            for res in valid_results:
                usage = res.get("llm_usage", {})
                if usage:
                    for key, value in usage.items():
                        if not value:
                            continue
                        if isinstance(value, list):
                            if key not in total_llm_usage:
                                total_llm_usage[key] = []
                            total_llm_usage[key].extend(value)
                        elif isinstance(value, (int, float)):
                            total_llm_usage[key] = total_llm_usage.get(key, 0) + value

            self.metrics_collector.record_metric(
                "mas_arena.problem.result",
                1 if is_correct else 0,
                {
                    "problem_id": problem_id,
                    "correct": is_correct,
                    "duration_ms": total_duration,
                    "pass_at_k": k,
                },
            )

            result_entry = {
                "problem_id": problem_id,
                "problem": normalized_problem["problem"],
                "expected": normalized_problem["solution"],
                "prediction": representative_result.get("extracted_answer") or representative_result.get("final_answer") or "",
                "score": 1 if is_correct else 0,
                "is_correct": is_correct,
                "status": "completed" if valid_results else "error",
                "reasoning": representative_result.get("reasoning", ""),
                "duration_ms": total_duration,
                "agent_system": agent_system,
                "llm_usage": total_llm_usage,
                "summary": {
                    "correct": is_correct,
                    "score": 1 if is_correct else 0,
                    "duration_ms": total_duration,
                },
                "pass_at_k_results": [str(e) if isinstance(e, Exception) else e for e in results_list],
            }

            if verbose:
                status_char = "✓" if is_correct else "✗"
                print(f"Result: {status_char} (total {total_duration:.0f}ms for {k} runs)")

            return result_entry
        except Exception as e:
            self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")
            self.metrics_collector.record_error(
                "problem_processing",
                str(e),
                {"problem_id": problem_id, "error_type": type(e).__name__},
            )
            if verbose:
                print(f"Error processing problem {problem_id}: {e}")
                traceback.print_exc()
            return {
                "problem_id": problem_id,
                "problem": normalized_problem.get("problem"),
                "status": "error",
                "error": str(e),
                "score": 0,
            }

    def _finalize_benchmark(
        self,
        all_results,
        benchmark_name,
        agent_system,
        output_file,
        verbose,
        pass_at_k=1,
        start_time=None,
    ):
        print(f"All results: {all_results}")
        run_end_time = time.perf_counter()
        wall_clock_time_ms = (run_end_time - start_time) * 1000 if start_time is not None else None

        total = len(all_results)
        if total == 0:
            print("No results to finalize.")
            return {}

        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        errored = sum(1 for r in all_results if r.get("status") == "error")

        # Calculate accuracy only on non-errored problems
        valid_runs = total - errored
        accuracy = correct / valid_runs if valid_runs > 0 else 0.0

        accuracy_metric_name = f"pass_at_{pass_at_k}_accuracy" if pass_at_k > 1 else "accuracy"

        total_duration = sum(r.get("duration_ms", 0) for r in all_results)

        # Calculate avg_tokens only on successful (non-errored) runs
        successful_runs = [r for r in all_results if r.get("status") != "error"]
        total_input_tokens = sum(
            sum(agent.get("prompt_tokens", 0) for agent in r.get("llm_usage", {}).get("agent_usage", []))
            for r in all_results
        )
        total_output_tokens = sum(
            sum(agent.get("completion_tokens", 0) for agent in r.get("llm_usage", {}).get("agent_usage", []))
            for r in all_results
        )
        total_tokens_successful = sum(r.get("llm_usage", {}).get("total_tokens", 0) for r in successful_runs)

        # Avoid division by zero: compute averages only when there are successful runs
        avg_input_tokens = total_input_tokens / len(successful_runs) if successful_runs else 0
        avg_output_tokens = total_output_tokens / len(successful_runs) if successful_runs else 0
        avg_tokens = total_tokens_successful / len(successful_runs) if successful_runs else 0

        summary = {
            "benchmark": benchmark_name,
            "agent_system": agent_system,
            "total_problems": total,
            "correct": correct,
            "errored": errored,
            accuracy_metric_name: accuracy,
            "total_duration_ms": total_duration,
            "wall_clock_time_ms": wall_clock_time_ms,
            "avg_duration_ms": total_duration / total if total > 0 else 0,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_tokens_per_successful_problem": avg_tokens,
            "results_file": str(output_file),
            # "metrics_dir": str(metrics_output),
            "timestamp": self.timestamp,
        }

        self.metrics_registry.stop_all_collectors()
        # self.metrics_registry.export_all(format="json", path=str(metrics_output))

        # Create a separate summary file for visualization
        summary_file = output_file.with_name(f"{output_file.stem}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4, default=custom_json_serializer)

        # Save main results file
        with open(output_file, "w") as f:
            json.dump(
                {"summary": summary, "results": all_results},
                f,
                indent=4,
                default=custom_json_serializer,
            )

        if verbose:
            # Run failure attribution analysis for incorrect results
            self._run_failure_attribution(all_results, agent_system, verbose)
            # Print benchmark summary
            print("\n" + "=" * 80)
            rprint("[bold green]🎯 Benchmark Summary[/bold green]")
            print("=" * 80)
            print(json.dumps(summary, indent=2))
            print("-" * 80)
            # Print visualization command
            rprint("[bold blue]📈 To visualize results, run:[/bold blue]")
            print(f"$ python mas_arena/visualization/visualize_benchmark.py visualize --summary {summary_file}")
            print("=" * 80)

        self.results = all_results
        self.summary = summary
        return summary

    def _collect_failed_responses(self, all_results, agent_system, verbose):
        """
        Collect and move failed agent response files to a centralized directory.

        Args:
            all_results: List of all benchmark results
            agent_system: Name of the agent system used
            verbose: Whether to print progress information

        Returns:
            Path to the directory containing failed responses, or None if no failures
        """
        # Find incorrect results
        incorrect_results = [r for r in all_results if r.get("score", 0) != 1 and "error" not in r]

        if not incorrect_results:
            return None

        # Find corresponding agent response files
        agent_responses_dir = Path("results/agent_responses")
        if not agent_responses_dir.exists():
            if verbose:
                print("Agent responses directory not found.")
            return None

        # Create centralized failed responses directory
        failed_responses_dir = Path(self.results_dir) / "failure" / f"failed_responses_{self.timestamp}"
        failed_responses_dir.mkdir(parents=True, exist_ok=True)

        collected_files = []

        for result in incorrect_results:
            problem_id = result.get("problem_id")
            if not problem_id:
                continue

            # Find the corresponding agent response file
            # Pattern: {agent_system}_{problem_id}_{timestamp}_{hash}.json
            pattern = f"{agent_system}_{problem_id}_*.json"
            matching_files = list(agent_responses_dir.glob(pattern))

            if not matching_files:
                if verbose:
                    print(f"No agent response file found for problem {problem_id}")
                continue

            # Use the most recent file if multiple matches
            agent_response_file = max(matching_files, key=lambda f: f.stat().st_mtime)

            # Copy file to failed responses directory
            dest_file = failed_responses_dir / agent_response_file.name
            shutil.copy2(agent_response_file, dest_file)
            collected_files.append(dest_file)

            if verbose:
                print(f"Collected failed response for problem {problem_id}: {agent_response_file.name}")

        if verbose:
            print(f"\nCollected {len(collected_files)} failed response files in: {failed_responses_dir}")

        return failed_responses_dir

    def _run_failure_attribution(self, all_results, agent_system, verbose):
        """
        Prepare and display instructions for failure attribution analysis.

        Args:
            all_results: List of all benchmark results
            agent_system: Name of the agent system used
            verbose: Whether to print progress information
        """
        # Find incorrect results
        incorrect_results = [r for r in all_results if r.get("score", 0) != 1 and "error" not in r]

        if not incorrect_results:
            if verbose:
                print("\n" + "=" * 80)
                rprint("[yellow]No incorrect results found. Skipping failure attribution analysis.[/yellow]")
                print("=" * 80)
            return

        if verbose:
            print("\n" + "=" * 80)
            rprint(
                f"[bold yellow]Found {len(incorrect_results)} incorrect results for Failure Attribution Analysis[/bold yellow]"
            )
            print("=" * 80)

        # Collect failed response files
        failed_responses_dir = self._collect_failed_responses(all_results, agent_system, verbose)

        if not failed_responses_dir:
            if verbose:
                print("Could not collect failed response files.")
            return

        # Check if failure inference script exists
        failure_inference_script = Path("mas_arena/failure/inference.py")
        if not failure_inference_script.exists():
            if verbose:
                print("Failure inference script not found.")
            return

        # Create failure output directory
        failure_output_dir = Path(self.results_dir) / "failure"
        failure_output_dir.mkdir(exist_ok=True)

        if verbose:
            print("\n" + "-" * 80)
            rprint("[bold blue]🔍 To run Failure Attribution Analysis, execute the following command:[/bold blue]")
            print("-" * 80)
            print(f"python {failure_inference_script} \\")
            print("    --method binary_search \\")
            print("    --model gpt-4.1 \\")
            print(f"    --directory_path {failed_responses_dir} \\")
            print(f"    --output_dir {failure_output_dir}")
            # print("-" * 80)
            rprint("\n[bold]Alternative analysis methods:[/bold]")
            print("#\n For comprehensive analysis:")
            print(
                f"python {failure_inference_script} --method all_at_once --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}"
            )
            print("#\n For efficient error localization in long conversations:")
            print(
                f"python {failure_inference_script} --method binary_search --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}"
            )
            print("\n# For detailed incremental analysis:")
            print(
                f"python {failure_inference_script} --method step_by_step --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}"
            )

            print("=" * 80)

    def run(
        self,
        benchmark_name="math",
        data_path=None,
        limit=None,
        agent_system="single_agent",
        agent_config=None,
        verbose=True,
        data_id=None,
        memory_type=None,
        pass_at_k=1,
        log_file=None,
    ):
        """
        Run a benchmark sequentially. This is a wrapper around arun.
        """
        return asyncio.run(
            self.arun(
                benchmark_name=benchmark_name,
                data_path=data_path,
                limit=limit,
                agent_system=agent_system,
                agent_config=agent_config,
                verbose=verbose,
                data_id=data_id,
                concurrency=1,  # Run sequentially
                memory_type=memory_type,
                pass_at_k=pass_at_k,
                log_file=log_file,
            )
        )

    async def arun(
        self,
        benchmark_name="math",
        data_path=None,
        limit=None,
        agent_system="single_agent",
        agent_config=None,
        verbose=True,
        data_id=None,
        concurrency=10,
        memory_type=None,
        pass_at_k=1,
        log_file=None,
    ):
        self._setup_logging(log_file)
        # Prepare benchmark; we only need problems and config here
        problems, benchmark_config, output_file = self._prepare_benchmark(
            benchmark_name,
            data_path,
            limit,
            agent_system,
            agent_config,
            verbose,
            data_id,
            memory_type,
        )

        if verbose:
            print(f"Running {benchmark_name} benchmark asynchronously with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems with concurrency {concurrency}.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")
        run_start_time = time.perf_counter()

        semaphore = asyncio.Semaphore(concurrency)

        if pass_at_k > 1:

            async def process_pass_at_k_with_semaphore(i, p):
                async with semaphore:
                    return await self._process_one_problem_pass_at_k(
                        i,
                        p,
                        agent_system,
                        self.agent_config,
                        benchmark_config,
                        verbose,
                        pass_at_k,
                        memory_type=memory_type,
                    )

            tasks = [process_pass_at_k_with_semaphore(i, p) for i, p in enumerate(problems)]
        else:

            async def process_with_semaphore(i, p):
                async with semaphore:
                    # Create a fresh agent instance per problem to isolate state
                    agent = create_agent_system(agent_system, self.agent_config, memory_type=memory_type)
                    agent.set_metrics_registry(self.metrics_registry)
                    return await self._process_one_problem(i, p, agent, benchmark_config, verbose)

            tasks = [process_with_semaphore(i, p) for i, p in enumerate(problems)]

        all_results = await tqdm.gather(*tasks, desc="Processing Problems")

        # if hasattr(agent, "aclose"):
        #     await agent.aclose()

        return self._finalize_benchmark(
            all_results,
            benchmark_name,
            agent_system,
            output_file,
            verbose,
            pass_at_k,
            start_time=run_start_time,
        )

    def visualize_results(self, output_dir=None):
        """
        Generate visualizations from the benchmark results.

        Args:
            output_dir: Directory to save visualizations (defaults to metrics_dir/benchmark_timestamp/viz)
        """
        pass
