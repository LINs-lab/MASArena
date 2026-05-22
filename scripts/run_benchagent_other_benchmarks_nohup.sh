#!/bin/bash
# Run BenchAgent on all registered non-GAIA / non-HotpotQA benchmarks in the background.
#
# Usage:
#   ./scripts/run_benchagent_other_benchmarks_nohup.sh
#   MAX_LIMIT=400 ./scripts/run_benchagent_other_benchmarks_nohup.sh
#   BENCHMARKS_CSV="math,gsm8k,bbh" ./scripts/run_benchagent_other_benchmarks_nohup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

DEFAULT_BENCHMARKS=(
  aime
  bbh
  drop
  gsm8k
  humaneval
  ifeval
  math
  mbpp
  mmlu_pro
  swebench_lite
)

if [ -n "${BENCHMARKS_CSV:-}" ]; then
  IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARKS_CSV}"
else
  BENCHMARKS=("${DEFAULT_BENCHMARKS[@]}")
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
CONCURRENCY="${CONCURRENCY:-3}"
MAX_LIMIT="${MAX_LIMIT:-400}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_other_benchmarks}"
MANAGER_TOOLS="${MANAGER_TOOLS:-python_interpreter,final_answer}"
SEARCH_TOOLS="${SEARCH_TOOLS:-python_interpreter,final_answer}"
OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-600}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_other_benchmarks_${TIMESTAMP}.log}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [ -x ".venv/bin/python" ]; then
  RUNNER=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
  RUNNER=(python3)
else
  echo "Error: no Python runner found (.venv/bin/python / uv / python3)."
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ]; then
  echo "Warning: OPENAI_API_KEY is not set and .env is missing; the run may fail."
fi

export MODEL_NAME
export OPENAI_API_TIMEOUT
export PYTHONUNBUFFERED
export PYTHONIOENCODING
export RESULTS_DIR
export LOG_DIR
export LOG_FILE
export CONCURRENCY
export MAX_LIMIT
export MANAGER_TOOLS
export SEARCH_TOOLS
export BENCHMARKS_CSV="${BENCHMARKS_CSV:-}"
export TIMESTAMP

resolve_data_path() {
  case "$1" in
    swebench_lite)
      echo "data/swebench_lite_test.jsonl"
      ;;
    *)
      echo "data/${1}_test.jsonl"
      ;;
  esac
}

if [ "${1:-}" != "--worker" ]; then
  echo "====================================================="
  echo "Running BenchAgent non-GAIA / non-HotpotQA benchmarks (nohup)"
  echo "Benchmarks: ${BENCHMARKS[*]}"
  echo "Model: $MODEL_NAME"
  echo "Concurrency: $CONCURRENCY"
  echo "Max limit per benchmark: $MAX_LIMIT"
  echo "Manager tools: $MANAGER_TOOLS"
  echo "Search tools: $SEARCH_TOOLS"
  echo "Log file: $LOG_FILE"
  echo "====================================================="
  echo "Monitor with: tail -f $LOG_FILE"

  nohup bash "$0" --worker > "$LOG_FILE" 2>&1 &