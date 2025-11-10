# Usage

This guide explains how to run benchmarks and use the automated workflow optimizer with MASArena.

## Prerequisites

1.  **Install dependencies:**
    If you haven't already, install the required packages. We recommend using `uv`.
    ```bash
    uv sync
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the project root and set your OpenAI API key and desired model.
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    MODEL_NAME=gpt-4o-mini
    OPENAI_API_BASE=https://api.openai.com/v1
    ```

## Running Benchmarks

You can run benchmarks using the convenience shell script `run_benchmark.sh` (recommended) or by directly calling `main.py`.

### Using the Shell Script (`run_benchmark.sh`)

The `run_benchmark.sh` script is the simplest way to run evaluations.

**Syntax:**
```bash
# Usage: ./run_benchmark.sh [benchmark] [agent_system] [limit] [mcp_config] [concurrency] [optimizer]
./run_benchmark.sh math supervisor_mas 10
```

**Examples:**

```bash
# Run the 'math' benchmark on 10 problems with the 'supervisor_mas' agent system
./run_benchmark.sh math supervisor_mas 10

# Run the 'humaneval' benchmark asynchronously with a concurrency of 10
# The "" is a placeholder for the mcp_config argument.
./run_benchmark.sh humaneval single_agent 20 "" 10
```

## Automated Workflow Optimization (AFlow)

MASArena includes AFlow implementation, an automated optimizer for agent workflows. 

**Example:**
To run AFlow to optimize an agent for the `humaneval` benchmark, provide `aflow` as the optimizer argument to the shell script:

```bash
# The "" arguments are placeholders for mcp_config and concurrency.
./run_benchmark.sh humaneval single_agent 10 "" "" aflow
```

You can also specify the training and test set sizes for the optimizer. Note that when using the `aflow` optimizer, the number of problems is determined by `train_size` and `test_size`, and the `limit` argument is ignored for data selection.

**Example with custom training and test sizes:**

```bash
# Run AFlow with a training set of 30 and a test set of 15.
# The "" arguments are placeholders for mcp_config and concurrency.
# The limit argument (10) is ignored in this case.
./run_benchmark.sh humaneval single_agent 10 "" "" aflow 30 15
```

## Command-Line Arguments

Here are the most common arguments for `main.py`.

### Main Arguments

| Argument | Description | Default |
|---|---|---|
| `--benchmark` | The name of the benchmark to run. | `math` |
| `--agent-system` | The agent system to use for the benchmark. | `single_agent` |
| `--limit` | The maximum number of problems to evaluate. | `None` (all) |
| `--data` | Path to a custom benchmark data file (JSONL format). | `data/{benchmark}_test.jsonl` |
| `--results-dir` | Directory to store detailed JSON results. | `results/` |
| `--verbose` | Print progress information. | `True` |
| `--async-run` | Run the benchmark asynchronously for faster evaluation. | `False` |
| `--concurrency` | Set the concurrency level for asynchronous runs. | `10` |
| `--use-tools` | Enable the agent to use integrated tools (e.g., code interpreter). | `False` |
| `--use-mcp-tools` | Enable the agent to use tools via MCP. | `False` |
| `--mcp-config-file`| Path to the MCP server configuration file. Required for MCP tools. | `None` |
| `--data-id` | Data ID to use. | `None` |

### Optimizer Arguments

These arguments are used when running an optimizer like AFlow via `--run-optimizer`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--run-optimizer` | str | `None` | Specifies the optimizer to run. Use `aflow`. |
| `--graph_path` | str | `mas_arena/configs/aflow` | Path to the base AFlow graph configuration. |
| `--optimized_path` | str | `example/aflow/humaneval/optimization` | Path to save the optimized AFlow graph. |
| `--validation_rounds`| int | 1 | Number of validation rounds per optimization cycle. |
| `--eval_rounds` | int | 1 | Number of evaluation rounds per optimization cycle. |
| `--max_rounds` | int | 3 | Maximum number of optimization rounds. |
| `--train_size` | int | 40 | Size of the training set for evaluation. |
| `--test_size` | int | 20 | Size of the test set for evaluation. |

## Example Output

After a run, a summary is printed to the console:

```bash
================================================================================
Benchmark Summary
================================================================================
Agent system: swarm
Accuracy: 70.00% (7/10)
Total duration: 335125ms
Results saved to: results/math_swarm_20250616_203434.json
Summary saved to: results/math_swarm_20250616_203434_summary.json

Run visualization:
$ python mas_arena/visualization/visualize_benchmark.py visualize \
  --summary results/math_swarm_20250616_203434_summary.json
```
