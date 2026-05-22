#!/bin/bash
# Run BenchAgent on a tiny GAIA validation slice with GLM in the background.
# Uses CHATGPT_API_KEY / OPENAI_API_BASE from .env, same as the Qwen scripts.
#
# Usage:
#   ./scripts/run_gaia_benchagent_glm.sh
#   ./scripts/run_gaia_benchagent_glm.sh 2 1
#   LIMIT=4 CONCURRENCY=2 ./scripts/run_gaia_benchagent_glm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

LIMIT="${LIMIT:-${1:-2}}"
CONCURRENCY="${CONCURRENCY:-${2:-1}}"
DATA_PATH="${DATA_PATH:-data/gaia_validate_level1.jsonl}"
MODEL_NAME="${MODEL_NAME:-${GLM_MODEL:-glm-5}}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_gaia_glm_${TIMESTAMP}.log}"

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

if [ -z "${CHATGPT_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ]; then
  echo "Warning: CHATGPT_API_KEY/OPENAI_API_KEY is not set and .env is missing; the run may fail."
fi

export MODEL_NAME
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-600}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "====================================================="
echo "Running BenchAgent GAIA validation with GLM (nohup)"
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
  --benchmark gaia \
  --agent-system bench_agent \
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
