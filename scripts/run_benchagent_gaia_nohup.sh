#!/bin/bash
# Run BenchAgent on a small GAIA validation slice in the background.
#
# Usage:
#   ./scripts/run_benchagent_gaia_nohup.sh
#   ./scripts/run_benchagent_gaia_nohup.sh 10 2
#   LIMIT=12 CONCURRENCY=3 MODEL_NAME=qwen3-32b ./scripts/run_benchagent_gaia_nohup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT="${LIMIT:-${1:-10}}"
CONCURRENCY="${CONCURRENCY:-${2:-2}}"
DATA_PATH="${DATA_PATH:-data/gaia_validate_level1.jsonl}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_gaia_${TIMESTAMP}.log}"

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: data file not found: $DATA_PATH"
  exit 1
fi

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
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-600}"

echo "====================================================="
echo "Running BenchAgent GAIA validation (nohup)"
echo "Data: $DATA_PATH"
echo "Limit: $LIMIT"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL_NAME"
echo "Manager tools: $MANAGER_TOOLS"
echo "Search tools: $SEARCH_TOOLS"
echo "Log file: $LOG_FILE"
echo "====================================================="
echo "Monitor with: tail -f $LOG_FILE"

nohup "${RUNNER[@]}" main.py \
  --benchmark gaia \
  --agent-system bench_agent \
  --data "$DATA_PATH" \
  --limit "$LIMIT" \
  --results-dir "$RESULTS_DIR" \
  --manager-tools "$MANAGER_TOOLS" \
  --search-tools "$SEARCH_TOOLS" \
  --async-run \
  --concurrency "$CONCURRENCY" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started in background. PID: $PID"
echo "Done. Log: $LOG_FILE | Results: $RESULTS_DIR"
