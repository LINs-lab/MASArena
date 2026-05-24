#!/bin/bash
# Run a MAS agent on GAIA validation samples.
#
# Usage:
#   ./scripts/run_MAS_gaia_level.sh
#   ./scripts/run_MAS_gaia_level.sh <limit> <concurrency> <mas> <data_path>
#   ./scripts/run_MAS_gaia_level.sh 10 2 Jarvis data/gaia_validate_level1.jsonl
#   LIMIT=20 CONCURRENCY=6 AGENT_SYSTEM=evoagent ./scripts/run_MAS_gaia_level.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

BENCHMARK="${BENCHMARK:-gaia}"
LIMIT="${LIMIT:-${1:-0}}"
CONCURRENCY="${CONCURRENCY:-${2:-2}}"
AGENT_SYSTEM="${AGENT_SYSTEM:-${3:-Jarvis}}"
DATA_PATH="${DATA_PATH:-${4:-data/gaia_validate_level3.jsonl}}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/${AGENT_SYSTEM}_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${AGENT_SYSTEM}_gaia_${MODEL_NAME}_${TIMESTAMP}.log}"

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
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-300}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "====================================================="
echo "Running MAS GAIA benchmark"
echo "Benchmark: $BENCHMARK"
echo "Agent system: $AGENT_SYSTEM"
echo "Data: $DATA_PATH"
echo "Limit: $LIMIT"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL_NAME"
echo "Manager tools: $MANAGER_TOOLS"
echo "Search tools: $SEARCH_TOOLS"
echo "Log file: $LOG_FILE"
echo "====================================================="
echo "Monitor with: tail -f $LOG_FILE"

touch "$LOG_FILE"

nohup env PYTHONUNBUFFERED="$PYTHONUNBUFFERED" PYTHONIOENCODING="$PYTHONIOENCODING" "${RUNNER[@]}" main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_PATH" \
  --limit "$LIMIT" \
  --results-dir "$RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --manager-tools "$MANAGER_TOOLS" \
  --search-tools "$SEARCH_TOOLS" \
  --log-file "$LOG_FILE" \
  --async-run \
  --concurrency "$CONCURRENCY" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started in background. PID: $PID"
echo "Done. Log: $LOG_FILE | Results: $RESULTS_DIR"
