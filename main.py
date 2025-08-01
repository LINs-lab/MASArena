#!/usr/bin/env python3
import argparse
import datetime
import os
import sys
import time
import atexit
from pathlib import Path
import asyncio
import json
import logging
from typing import Dict, Any, Optional

from mas_arena.benchmark_runner import BenchmarkRunner
from mas_arena.tools.browser_cleanup import cleanup_browser_processes

logger = logging.getLogger(__name__)


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load MCP configuration from a JSON file.
    
    Args:
        config_path: Path to the MCP configuration file
        
    Returns:
        Dictionary containing MCP server configurations
    """
    if not config_path:
        # 默认路径
        default_paths = [
            "mas_arena/mcp_collections/mcp.json",
            "MASArena/mas_arena/mcp_collections/mcp.json"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"MCP configuration file not found at {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if "mcpServers" not in config:
            return {"mcpServers": config}
        else:
            return config
            
    except Exception as e:
        logger.error(f"Error loading MCP configuration: {e}")
        return {}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run benchmarks for multi-agent systems")

    # Import available agent systems and benchmarks
    from mas_arena.agents import AVAILABLE_AGENT_SYSTEMS
    from mas_arena.evaluators import BENCHMARKS
    parser.add_argument(
        "--benchmark",
        type=str,
        default="math",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run (default: math)",
    )

    parser.add_argument(
        "--data", type=str, default=None, help="Path to benchmark data (default: data/{benchmark}_test.jsonl)"
    )

    parser.add_argument("--limit", type=int, default=None, help="Maximum number of problems to process (default: None)")

    parser.add_argument(
        "--agent-system",
        type=str,
        default="single_agent",
        choices=list(AVAILABLE_AGENT_SYSTEMS.keys()),
        help="Agent system to use (default: single_agent)",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print progress information (default: True)"
    )

    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to store results (default: results)"
    )

    parser.add_argument(
        "--use-mcp-tools", action="store_true", default=False,
        help="Enable integration of MCP tools (default: False)"
    )

    parser.add_argument(
        "--mcp-config-file", type=str, default=None,
        help="Path to MCP servers configuration JSON file"
    )

    parser.add_argument(
        "--async-run", action="store_true", help="Run the benchmark asynchronously."
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrency level for async run."
    )

    parser.add_argument(
        "--data-id", type=str, default=None,
        help="Data ID to use (default: None)"
    )

    # Optimizer arguments
    optimizer_group = parser.add_argument_group("Optimizer Settings")
    optimizer_group.add_argument(
        "--run-optimizer",
        type=str,
        default=None,
        choices=["aflow"],
        help="Run an optimization process instead of a benchmark.",
    )
    optimizer_group.add_argument(
        "--graph_path",
        type=str,
        default="mas_arena/configs/aflow",
        help="Path to the agent flow graph configuration.",
    )
    optimizer_group.add_argument(
        "--optimized_path",
        type=str,
        default=None,
        help="Path to save the optimized agent flow graph.",
    )
    optimizer_group.add_argument("--validation_rounds", type=int, default=1, help="Number of validation rounds.")
    optimizer_group.add_argument("--eval_rounds", type=int, default=1, help="Number of evaluation rounds.")
    optimizer_group.add_argument("--max_rounds", type=int, default=3, help="Maximum number of optimization rounds.")
    optimizer_group.add_argument("--train_size", type=int, default=40, help="Size of the training set for evaluation.")
    optimizer_group.add_argument("--test_size", type=int, default=20, help="Size of the test set for evaluation.")

    # Parse arguments
    args = parser.parse_args()

    if args.run_optimizer:
        if args.run_optimizer == "aflow":
            if not args.optimized_path:
                args.optimized_path = f"example/aflow/{args.benchmark}/optimization"

            if os.path.exists(args.optimized_path):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                args.optimized_path = f"{args.optimized_path}_{timestamp}"

            from example.aflow.run_aflow_optimize import run_aflow_optimization
            print("\n" + "=" * 80)
            print(f"Running AFlow Optimizer ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            print("=" * 80)
            print(f"Benchmark: {args.benchmark}")
            print(f"Graph Path: {args.graph_path}")
            print(f"Optimized Path: {args.optimized_path}")
            print("=" * 80 + "\n")
            
            # Run optimization and get the path to the final graph
            optimized_graph_path = run_aflow_optimization(args)
            
            # Set up to run the benchmark on the optimized agent
            args.agent_system = "single_agent"  # AFlow's executor is a single_agent
            args.agent_graph_config = optimized_graph_path

            print("\n" + "=" * 80)
            print("AFlow optimization finished. Now running benchmark on the optimized agent...")
            print(f"Optimized graph: {optimized_graph_path}")
            print("=" * 80 + "\n")
        else:
            print(f"Unknown optimizer: {args.run_optimizer}", file=sys.stderr)
            return 1
        # The script will now continue to the benchmark run part below
    
    # Build agent configuration for MCP tool integration
    agent_config = {}
    if args.use_mcp_tools:
        agent_config["use_mcp_tools"] = True
        
        if not args.mcp_config_file:
            mcp_servers = load_mcp_config()
        else:
            mcp_servers = load_mcp_config(args.mcp_config_file)
            
        agent_config["mcp_servers"] = mcp_servers
        
        agent_config["mcp_config_file"] = args.mcp_config_file

    # Create directories if needed
    Path(args.results_dir).mkdir(exist_ok=True)

    # Print header
    
    print("\n" + "=" * 80)
    print(f"Multi-Agent Benchmark Runner ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)
    print(f"Benchmark: {args.benchmark}")
    print(f"Agent System: {args.agent_system}")
    print(f"Data: {args.data or 'default'}")
    print(f"Limit: {args.limit or 'all'}")
    print("=" * 80 + "\n")

    # Create benchmark runner
    runner = BenchmarkRunner(results_dir=args.results_dir)

    # Check for concurrency support
    benchmark_config = BENCHMARKS.get(args.benchmark, {})
    evaluator_class = benchmark_config.get("evaluator")
    supports_concurrency = evaluator_class and getattr(evaluator_class, 'SUPPORTS_CONCURRENCY', True)

    run_async = args.async_run and supports_concurrency
    if args.async_run and not supports_concurrency:
        if args.verbose:
            print(f"Warning: {args.benchmark} benchmark does not support concurrency. Running synchronously.\n")

    # Run benchmark
    try:
        if run_async:
            summary = asyncio.run(runner.arun(
                benchmark_name=args.benchmark,
                data_path=args.data,
                limit=args.limit,
                agent_system=args.agent_system,
                agent_config=agent_config if agent_config else None,
                verbose=args.verbose,
                data_id=args.data_id,
                concurrency=args.concurrency,
            ))
        else:
            summary = runner.run(
                benchmark_name=args.benchmark,
                data_path=args.data,
                limit=args.limit,
                agent_system=args.agent_system,
                agent_config=agent_config if agent_config else None,
                verbose=args.verbose,
                data_id=args.data_id,
            )
        logger.info(f"Benchmark summary: {summary}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Register browser cleanup function to run at exit
def cleanup_at_exit():
    print("\nPerforming final cleanup of MCP browser processes...")
    # Use the more aggressive approach to ensure all MCP-related processes are terminated
    from mas_arena.tools.browser_cleanup import kill_mcp_browser_processes
    kill_mcp_browser_processes(verbose=True)
    # Then use the regular cleanup for any remaining processes and temp directories
    cleanup_browser_processes(verbose=True, force=True, cleanup_temp=True, mcp_only=True)
    print("Cleanup complete.")

atexit.register(cleanup_at_exit)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
        cleanup_browser_processes(verbose=True, force=True, cleanup_temp=True, mcp_only=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nProgram encountered an error: {e}")
        print("Cleaning up MCP browser processes before exit...")
        cleanup_browser_processes(verbose=True, force=True, cleanup_temp=True, mcp_only=True)
        raise
