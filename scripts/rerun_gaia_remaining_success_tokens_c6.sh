#!/bin/bash
# Rerun GAIA remaining problems plus success-token补跑集 with BenchAgent.
#
# Usage:
#   ./scripts/rerun_gaia_remaining_success_tokens_c6.sh
#   CONCURRENCY=3 ./scripts/rerun_gaia_remaining_success_tokens_c6.sh
#   DATA_PATH=data/custom.jsonl LOG_FILE=logs/bench_agent_gaia/custom.log ./scripts/rerun_gaia_remaining_success_tokens_c6.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

DATA_PATH="${DATA_PATH:-data/gaia_validate_level2_rerun_remaining_plus_success_tokens_20260522.jsonl}"
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-6}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_gaia_level2_rerun_remaining_success_tokens_c${CONCURRENCY}_${TIMESTAMP}.log}"

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

export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-600}"
export CHATGPT_KEY_ROTATION_ENABLED="${CHATGPT_KEY_ROTATION_ENABLED:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "====================================================="
echo "Rerunning GAIA remaining + success-token补跑 (nohup)"
echo "Data: $DATA_PATH"
echo "Limit: $LIMIT"
echo "Concurrency: $CONCURRENCY"
echo "ChatGPT key rotation: $CHATGPT_KEY_ROTATION_ENABLED"
echo "Manager tools: $MANAGER_TOOLS"
echo "Search tools: $SEARCH_TOOLS"
echo "Log file: $LOG_FILE"
echo "====================================================="
echo "Monitor with: tail -f $LOG_FILE"

touch "$LOG_FILE"

nohup env CHATGPT_KEY_ROTATION_ENABLED="$CHATGPT_KEY_ROTATION_ENABLED" PYTHONUNBUFFERED="$PYTHONUNBUFFERED" PYTHONIOENCODING="$PYTHONIOENCODING" "${RUNNER[@]}" main.py \
  --benchmark gaia \
  --agent-system bench_agent \
  --data "$DATA_PATH" \
  --limit "$LIMIT" \
  --results-dir "$RESULTS_DIR" \
  --manager-tools "$MANAGER_TOOLS" \
  --search-tools "$SEARCH_TOOLS" \
  --log-file "$LOG_FILE" \
  --async-run \
  --concurrency "$CONCURRENCY" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started in background. PID: $PID"
echo "Done. Log: $LOG_FILE | Results: $RESULTS_DIR"
