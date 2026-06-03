#!/bin/bash
#===============================================================================
# MAS Arena — Official Benchmark Entry Point
#===============================================================================
#
# Quick start (lazy mode):
#     ./run_benchmark.sh
#   Runs math benchmark with bench_agent using all defaults.
#
# Override via environment variables:
#     BENCHMARK=gaia AGENT_SYSTEM=evoagent CONCURRENCY=8 ./run_benchmark.sh
#     BENCHMARK=mmlu_pro LIMIT=50 MODEL_NAME=gpt-4o ./run_benchmark.sh
#
# All configuration lives in the env-var block below — nothing hardcoded.
# The script uses `uv run python` so you only need `uv` installed.
#===============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# --- Environment check --------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "[ERROR] uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

RUNNER=(uv run python)

if [ -f .env ]; then
  set -a; source .env; set +a
fi

#===============================================================================
# All configurable parameters (override via env vars)
#===============================================================================

# --- Benchmark ----------------------------------------------------------------
BENCHMARK="${BENCHMARK:-math}"
  # Benchmark to run. Available: math, gsm8k, humaneval, mbpp, hotpotqa,
  # gaia, drop, bbh, mmlu_pro, aime, swebench_lite, ifeval, musique.
DATA_PATH="${DATA_PATH:-}"
  # Path to data file. Defaults to data/{benchmark}_test.jsonl if empty.
  # Example: data/gaia_validate_level1.jsonl
LIMIT="${LIMIT:-0}"
  # Max problems to evaluate (0 = all).

# --- Agent system -------------------------------------------------------------
AGENT_SYSTEM="${AGENT_SYSTEM:-bench_agent}"
  # Agent system name. Available examples (may vary by installed plugins):
  # bench_agent, jarvis, evoagent, chateval, autogen, camel, llm_debate,
  # metagpt, single_agent, swarm, supervisor_mas, mad, chatdev.

# --- Model --------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
  # Model name passed through to the agent system.

# --- Concurrency --------------------------------------------------------------
ASYNC_RUN="${ASYNC_RUN:-true}"
  # Run benchmarks asynchronously (true / false).
CONCURRENCY="${CONCURRENCY:-4}"
  # Number of parallel workers when async is enabled.

# --- Tools --------------------------------------------------------------------
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
  # Comma-separated manager tools. "ALL" = use default set. "none" = no tools.
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
  # Comma-separated search tools. "ALL" = use default set. "none" = no tools.
USE_MCP="${USE_MCP:-false}"
  # Enable MCP tool integration (true / false).
MCP_CONFIG="${MCP_CONFIG:-}"
  # Path to MCP server configuration JSON file.

# --- Memory -------------------------------------------------------------------
MEMORY_TYPE="${MEMORY_TYPE:-}"
  # Memory system to enable (empty = disabled). Example: melo, memory_bank.

# --- Agent tuning knobs -------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-}"
  # Manager CodeAgent max steps (agent default: 15).
SEARCH_MAX_STEPS="${SEARCH_MAX_STEPS:-}"
  # Search ToolCallingAgent max steps (agent default: 10).
DEBATE_ROUNDS="${DEBATE_ROUNDS:-}"
  # Debate rounds for llm_debate agent (default: 3).
DEBATE_AGENTS="${DEBATE_AGENTS:-}"
  # Number of debate agents for llm_debate (default: 2).
NUM_ROUNDS="${NUM_ROUNDS:-}"
  # Debate rounds for chateval_newcore (default: 2).
PASS_AT_K="${PASS_AT_K:-1}"
  # Number of samples to generate for pass@k metric.

# --- Timeouts -----------------------------------------------------------------
PROBLEM_TIMEOUT="${PROBLEM_TIMEOUT:-}"
  # Hard timeout per problem in seconds (default: 3600).
EVO_TIMEOUT="${EVO_TIMEOUT:-}"
  # EvoAgent per-worker timeout in seconds (default: 300).
EVO_WORKER_DELAY="${EVO_WORKER_DELAY:-}"
  # Delay between starting EvoAgent workers in seconds.

# --- Optimizer (experimental) -------------------------------------------------
OPTIMIZER="${OPTIMIZER:-}"
  # Optimizer to run instead of plain benchmark. Currently: aflow.
TRAIN_SIZE="${TRAIN_SIZE:-}"
  # Training set size for optimizer evaluation.
TEST_SIZE="${TEST_SIZE:-}"
  # Test set size for optimizer evaluation.
GRAPH_PATH="${GRAPH_PATH:-}"
  # Path to agent flow graph configuration.
OPTIMIZED_PATH="${OPTIMIZED_PATH:-}"
  # Path to save optimized agent flow graph.
MAX_ROUNDS="${MAX_ROUNDS:-}"
  # Maximum optimization rounds.
VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-}"
  # Number of validation rounds for optimizer.
EVAL_ROUNDS="${EVAL_ROUNDS:-}"
  # Number of evaluation rounds for optimizer.

# --- Output -------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-results}"
  # Directory to store results.
LOG_DIR="${LOG_DIR:-logs}"
  # Directory to store log files.
VERBOSE="${VERBOSE:-true}"
  # Print progress information (true / false).

#===============================================================================
# Preflight
#===============================================================================

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Default data path
if [ -z "$DATA_PATH" ]; then
  DATA_PATH="data/${BENCHMARK}_test.jsonl"
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "[ERROR] Data file not found: $DATA_PATH"
  echo "  Available data files:"
  ls data/*.jsonl 2>/dev/null | head -20 || echo "  (no .jsonl files in data/)"
  exit 1
fi

LOG_FILE="${LOG_FILE:-$LOG_DIR/${AGENT_SYSTEM}_${BENCHMARK}_${MODEL_NAME}_${TIMESTAMP}.log}"

#===============================================================================
# Build argument list for main.py
#===============================================================================

ARGS=(
  --benchmark "$BENCHMARK"
  --agent-system "$AGENT_SYSTEM"
  --data "$DATA_PATH"
  --results-dir "$RESULTS_DIR"
  --model-name "$MODEL_NAME"
  --pass_at_k "$PASS_AT_K"
)

# Limit
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  ARGS+=(--limit "$LIMIT")
fi

# Concurrency
if [ "$ASYNC_RUN" = "true" ]; then
  ARGS+=(--async-run --concurrency "$CONCURRENCY")
fi

# Tools
if [ -n "$MANAGER_TOOLS" ] && [ "$MANAGER_TOOLS" != "ALL" ]; then
  ARGS+=(--manager-tools "$MANAGER_TOOLS")
fi
if [ -n "$SEARCH_TOOLS" ] && [ "$SEARCH_TOOLS" != "ALL" ]; then
  ARGS+=(--search-tools "$SEARCH_TOOLS")
fi
if [ "$USE_MCP" = "true" ] && [ -n "$MCP_CONFIG" ]; then
  ARGS+=(--use-mcp-tools --mcp-config-file "$MCP_CONFIG")
fi

# Memory
if [ -n "$MEMORY_TYPE" ]; then
  ARGS+=(--memory-type "$MEMORY_TYPE")
fi

# Agent tuning
[ -n "$MAX_STEPS" ]         && ARGS+=(--max-steps "$MAX_STEPS")
[ -n "$SEARCH_MAX_STEPS" ]  && ARGS+=(--search-max-steps "$SEARCH_MAX_STEPS")
[ -n "$DEBATE_ROUNDS" ]     && ARGS+=(--debate-rounds "$DEBATE_ROUNDS")
[ -n "$DEBATE_AGENTS" ]     && ARGS+=(--debate-agents "$DEBATE_AGENTS")
[ -n "$NUM_ROUNDS" ]        && ARGS+=(--num-rounds "$NUM_ROUNDS")

# Timeouts
[ -n "$PROBLEM_TIMEOUT" ]   && ARGS+=(--problem-timeout-seconds "$PROBLEM_TIMEOUT")
[ -n "$EVO_TIMEOUT" ]       && ARGS+=(--evo-agent-timeout-seconds "$EVO_TIMEOUT")
[ -n "$EVO_WORKER_DELAY" ]  && ARGS+=(--evo-worker-delay-seconds "$EVO_WORKER_DELAY")

# Logging
[ -n "$LOG_FILE" ] && ARGS+=(--log-file "$LOG_FILE")

# Optimizer
if [ -n "$OPTIMIZER" ]; then
  ARGS+=(--run-optimizer "$OPTIMIZER")
  [ -n "$TRAIN_SIZE" ]        && ARGS+=(--train_size "$TRAIN_SIZE")
  [ -n "$TEST_SIZE" ]         && ARGS+=(--test_size "$TEST_SIZE")
  [ -n "$GRAPH_PATH" ]        && ARGS+=(--graph_path "$GRAPH_PATH")
  [ -n "$OPTIMIZED_PATH" ]    && ARGS+=(--optimized_path "$OPTIMIZED_PATH")
  [ -n "$MAX_ROUNDS" ]        && ARGS+=(--max_rounds "$MAX_ROUNDS")
  [ -n "$VALIDATION_ROUNDS" ] && ARGS+=(--validation_rounds "$VALIDATION_ROUNDS")
  [ -n "$EVAL_ROUNDS" ]       && ARGS+=(--eval_rounds "$EVAL_ROUNDS")
fi

#===============================================================================
# Banner
#===============================================================================

echo "======================================================================"
echo "  MAS Arena Benchmark Runner"
echo "======================================================================"
echo "  Benchmark:       $BENCHMARK"
echo "  Agent System:    $AGENT_SYSTEM"
echo "  Model:           $MODEL_NAME"
echo "  Data:            $DATA_PATH"
echo "  Limit:           $([ "$LIMIT" -gt 0 ] 2>/dev/null && echo "$LIMIT" || echo "all")"
echo "  Concurrency:     $([ "$ASYNC_RUN" = "true" ] && echo "$CONCURRENCY (async)" || echo "off (sync)")"
echo "  Memory:          ${MEMORY_TYPE:-none}"
echo "  Pass@k:          $PASS_AT_K"
echo "  Log:             $LOG_FILE"
echo "  Results:         $RESULTS_DIR"
echo "======================================================================"
echo "  Monitor: tail -f $LOG_FILE"
echo "======================================================================"
echo ""

#===============================================================================
# Run (foreground — the user can Ctrl-C)
#===============================================================================

touch "$LOG_FILE"

exec env PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  "${RUNNER[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
