#!/usr/bin/env python3
"""
Simple Benchmark Runner Interface

This module provides a simplified interface for running benchmarks on multi-agent systems.
"""

import os
import json
import random
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from openai.types.completion_usage import CompletionUsage
import traceback
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint

from mas_arena.metrics import (
    MetricsRegistry,
    MetricsCollector
)
from mas_arena.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.utils.normalization import normalize_problem_keys

def custom_json_serializer(obj):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if hasattr(obj, '__dict__'):
        # For AIMessage-like objects, convert to dict
        return {key: getattr(obj, key) for key in obj.__dict__ if not key.startswith('_')}
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

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        # os.makedirs(metrics_dir, exist_ok=True)

        # Set up metrics
        self.metrics_registry = self._setup_metrics()
        
        # Create centralized metrics collector
        self.metrics_collector = MetricsCollector(self.metrics_registry)

    def _setup_metrics(self):
        """Set up metrics collection"""
        registry = MetricsRegistry()
        return registry

    def _prepare_benchmark(self, benchmark_name, data_path, limit, agent_system, agent_config, verbose):
        """
        Run a benchmark with the specified configuration.

        Args:
            benchmark_name: Name of the benchmark (math, drop, gsm8k, hotpotqa, humaneval, mbpp)
            data_path: Custom path to benchmark data file (optional)
            limit: Maximum number of problems to process
            agent_system: Agent system to use (single_agent, supervisor_mas, swarm)
            agent_config: Configuration for the agent system
            verbose: Whether to print progress information

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
        self.agent_config['evaluator'] = benchmark_name

        if not data_path:
            data_path = benchmark_config.get("data_path", f"data/{benchmark_name}_test.jsonl")

        output_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}.json"

        agent = create_agent_system(agent_system, self.agent_config)
        if agent is None:
            raise ValueError(f"Unknown agent system: {agent_system}. Available: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")

        agent.set_metrics_registry(self.metrics_registry)

        try:
            with open(data_path, "r") as f:
                problems = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if limit and limit < len(problems):
            problems = random.sample(problems, limit)

        return agent, problems, benchmark_config, output_file

    async def _process_one_problem(self, i, p, agent, benchmark_config, verbose=True):
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(p, key_mapping, i)
        problem_id = normalized_problem["id"]

        if verbose:
            print(f"Problem {i + 1}: {problem_id}")

        self.metrics_collector.start_timer(f"mas_arena.problem.{problem_id}", {"problem_id": problem_id})

        try:
            results = await agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)
            problem_duration_ms = self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")

            duration_ms = results.get("execution_time_ms", problem_duration_ms)
            score = results.get("score", 0)
            is_correct = results.get("is_correct", score == 1)

            self.metrics_collector.record_metric("mas_arena.problem.result", score, {"problem_id": problem_id, "correct": is_correct, "duration_ms": duration_ms})

            result_entry = {
                "problem_id": problem_id,
                "problem": normalized_problem["problem"],
                "expected": normalized_problem["solution"],
                "prediction": results.get("extracted_answer", ""),
                "score": score,
                "is_correct": is_correct,
                "status": results.get("status"),
                "reasoning": results.get("reasoning", ""),
                "duration_ms": duration_ms,
                "agent_system": agent.name,
                "llm_usage": results.get("llm_usage", {}),
                "summary": {"correct": is_correct, "score": score, "duration_ms": duration_ms},
            }
            if verbose:
                status_char = "E" if results.get("status") == "error" else "✓" if is_correct else "✗"
                print(f"Result: {status_char} ({duration_ms:.0f}ms)")
            return result_entry
        except Exception as e:
            self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")
            self.metrics_collector.record_error("problem_processing", str(e), {"problem_id": problem_id, "error_type": type(e).__name__})
            if verbose:
                print(f"Error processing problem {problem_id}: {e}")
                traceback.print_exc()
            return {"problem_id": problem_id, "problem": normalized_problem.get("problem"), "status": "error", "error": str(e), "score": 0}

    def _finalize_benchmark(self, all_results, benchmark_name, agent_system, output_file, verbose):
        total = len(all_results)
        if total == 0:
            print("No results to finalize.")
            return {}

        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        errored = sum(1 for r in all_results if r.get("status") == "error")
        
        # Calculate accuracy only on non-errored problems
        valid_runs = total - errored
        accuracy = correct / valid_runs if valid_runs > 0 else 0

        total_duration = sum(r.get("duration_ms", 0) for r in all_results)
        
        # Calculate avg_tokens only on successful (non-errored) runs
        successful_runs = [r for r in all_results if r.get("status") != "error"]
        total_tokens_successful = sum(r.get("llm_usage", {}).get("total_tokens", 0) for r in successful_runs)
        avg_tokens = total_tokens_successful / len(successful_runs) if successful_runs else 0

        summary = {
            "benchmark": benchmark_name,
            "agent_system": agent_system,
            "total_problems": total,
            "correct": correct,
            "errored": errored,
            "accuracy": accuracy,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / total if total > 0 else 0,
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
            json.dump({"summary": summary, "results": all_results}, f, indent=4, default=custom_json_serializer)
       
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
            rprint(f"[bold yellow]Found {len(incorrect_results)} incorrect results for Failure Attribution Analysis[/bold yellow]")
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
            print(f"    --method binary_search \\")
            print(f"    --model gpt-4.1 \\")
            print(f"    --directory_path {failed_responses_dir} \\")
            print(f"    --output_dir {failure_output_dir}")
            # print("-" * 80)
            rprint("\n[bold]Alternative analysis methods:[/bold]")
            print(f"# For comprehensive analysis:")
            print(f"python {failure_inference_script} --method binary_search --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}")
            print(f"\n# For step-by-step analysis:")
            print(f"python {failure_inference_script} --method step_by_step --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}")

            print("=" * 80)

    def run(self, benchmark_name="math", data_path=None, limit=None, agent_system="single_agent", agent_config=None, verbose=True):
        agent, problems, benchmark_config, output_file = self._prepare_benchmark(
            benchmark_name, data_path, limit, agent_system, agent_config, verbose
        )

        if verbose:
            print(f"Running {benchmark_name} benchmark with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems sequentially.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")

        # Since _process_one_problem is now async, we need to run it in an event loop
        # For sequential execution, we can still use an event loop to run one at a time
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        all_results = [
            loop.run_until_complete(self._process_one_problem(i, p, agent, benchmark_config, verbose))
            for i, p in enumerate(tqdm(problems, desc="Processing Problems"))
        ]

        return self._finalize_benchmark(all_results, benchmark_name, agent_system, output_file, verbose)

    async def arun(self, benchmark_name="math", data_path=None, limit=None, agent_system="single_agent", agent_config=None, verbose=True, concurrency=10):
        agent, problems, benchmark_config, output_file = self._prepare_benchmark(
            benchmark_name, data_path, limit, agent_system, agent_config, verbose
        )

        if verbose:
            print(f"Running {benchmark_name} benchmark asynchronously with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems with concurrency {concurrency}.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")

        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(i, p):
            async with semaphore:
                return await self._process_one_problem(i, p, agent, benchmark_config, verbose)

        tasks = [
            process_with_semaphore(i, p)
            for i, p in enumerate(problems)
        ]
        
        all_results = await tqdm.gather(*tasks, desc="Processing Problems")

        return self._finalize_benchmark(all_results, benchmark_name, agent_system, output_file, verbose)

    def visualize_results(self, output_dir=None):
        """
        Generate visualizations from the benchmark results.

        Args:
            output_dir: Directory to save visualizations (defaults to metrics_dir/benchmark_timestamp/viz)
        """
        pass


