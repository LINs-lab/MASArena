# AFlowOptimizer User Guide

## Introduction

AFlowOptimizer is a core component of the MASArena framework for automated optimization of multi-agent workflows. It leverages LLM-driven evolutionary optimization to automatically modify and evaluate workflow code, aiming to improve performance on a specified benchmark.

AFlow supports multi-round iterative optimization. In each round, it generates new workflow variants based on historical performance, validates them on evaluation sets, and selects the best-performing solution. The final optimized agent is then evaluated using the standard `BenchmarkRunner`, ensuring consistent metrics and enabling access to visualization and failure analysis tools.

---

## Key Features

- **Automated Evolutionary Optimization**: Uses LLM feedback to automatically modify workflow structure and prompts.
- **Multi-round Iteration**: Supports multiple optimization rounds and convergence checks.
- **Integrated Evaluation**: Optimized agents are evaluated through the standard `BenchmarkRunner` for consistent results.
- **Benchmark Agnostic**: Works with various benchmarks (e.g., humaneval, math).
- **Highly Extensible**: Supports custom operators, agents, and evaluators.

---

## Quick Start

### 1. Environment Setup

- Ensure you have set the following environment variables (e.g., in a `.env` file):
  - `OPENAI_API_KEY`
  - `OPENAI_API_BASE`
  - (Optional) `OPTIMIZER_MODEL_NAME`, `EXECUTOR_MODEL_NAME`

### 2. Run Optimization and Evaluation

The optimization process is now integrated into the main benchmark runner. Use the `run_benchmark.sh` script and specify an optimizer as the last argument.

```bash
# General usage
./run_benchmark.sh [benchmark] [agent_system] [limit] [mcp_config] [concurrency] [optimizer]

# Example: Run AFlow optimization on humaneval
./run_benchmark.sh humaneval single_agent 10 "" 1 aflow
```

This command will:
1.  Run the AFlow optimization process for the `humaneval` benchmark.
2.  Once optimization is complete, it will automatically run a standard benchmark evaluation on the newly optimized agent.
3.  The results will be saved in the `results/` directory, compatible with visualization and failure analysis tools.

---

## Main Script Arguments

The following arguments in `main.py` control the optimization process.

| Argument                | Type   | Default                                | Description                                                  |
|-------------------------|--------|----------------------------------------|--------------------------------------------------------------|
| `--run-optimizer`       | str    | `None`                                 | Specifies the optimizer to run. Use `aflow`.                 |
| `--benchmark`           | str    | `humaneval`                            | Benchmark to optimize for.                                   |
| `--graph_path`          | str    | `mas_arena/configs/aflow`              | Path to the base AFlow graph configuration.                  |
| `--optimized_path`      | str    | `example/aflow/humaneval/optimization` | Path to save the optimized AFlow graph and intermediate files. |
| `--validation_rounds`   | int    | 1                                      | Number of validation rounds per optimization cycle.          |
| `--eval_rounds`         | int    | 1                                      | Number of evaluation rounds per optimization cycle.          |
| `--max_rounds`          | int    | 3                                      | Maximum number of optimization rounds.                       |

---

## Example Standalone Usage (Advanced)

While the integrated workflow is recommended, you can run the optimization process standalone by executing `example/aflow/run_aflow_optimize.py`. This will only generate the optimized graph without running the final evaluation.

```python
# This example is simplified from example/aflow/run_aflow_optimize.py
import os
from dotenv import load_dotenv
from mas_arena.agents import AgentSystemRegistry
from mas_arena.evaluators import BENCHMARKS
from mas_arena.optimizers.aflow.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG

# --- Configuration ---
BENCHMARK_NAME = "humaneval"
# ... (load env vars and models) ...

# --- Initialization ---
# ... (initialize optimizer_agent, executor_agent, evaluator) ...

# --- Optimizer Setup ---
optimizer = AFlowOptimizer(
    # ... (optimizer parameters) ...
)

# --- Run Optimization ---
optimizer.setup()
optimizer.optimize(evaluator)
# The optimized graph is saved in your optimized_path

# To evaluate, you must then run the main benchmark script:
# python main.py --benchmark humaneval --agent-system single_agent --agent-graph-config path/to/your/final_graph.json
```

---

## Integrated Workflow

1.  **Trigger**: The user runs `main.py` with `--run-optimizer aflow`.
2.  **Optimization**: The `AFlowOptimizer` is invoked. It iteratively generates and evaluates workflow variants, producing a `final_graph.json` in the specified `optimized_path`.
3.  **Evaluation**: `main.py` automatically takes the path to `final_graph.json`.
4.  **Benchmark Run**: The `BenchmarkRunner` is called to execute a standard benchmark on a `single_agent` configured with the new optimized graph.
5.  **Results**: The results are saved in the standard format, making them available for all downstream analysis and visualization tools.

---

## FAQ

**Q: What models are used for optimization and execution?**
A: By default, `gpt-4o` for optimization and `gpt-4o-mini` for execution. You can override these via the `OPTIMIZER_MODEL_NAME` and `EXECUTOR_MODEL_NAME` environment variables.

**Q: How do I evaluate an optimized agent again later?**
A: Run the main benchmark script and point to the optimized graph file using the `--agent-graph-config` argument:
`python main.py --benchmark humaneval --agent-system single_agent --agent-graph-config path/to/final_graph.json`

**Q: Where are the optimized workflows saved?**
A: In the directory specified by `--optimized_path`. The final, best-performing graph is saved as `final_graph.json`.

---

## References
- See `run_benchmark.sh` and `main.py` for the primary usage pattern.
- See `example/aflow/run_aflow_optimize.py` for a reference on running standalone optimization.
- See `mas_arena/optimizers/aflow/aflow_optimizer.py` for the core optimizer implementation.
- See `mas_arena/optimizers/aflow/aflow_experimental_config.py` for benchmark-specific configurations.
