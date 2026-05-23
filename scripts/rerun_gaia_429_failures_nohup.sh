#!/bin/bash
# 统计并重跑 GAIA bench_agent 结果中因 429 / TPM 限流失败的任务（后台 nohup）。
#
# Usage:
#   ./scripts/rerun_gaia_429_failures_nohup.sh
#   ./scripts/rerun_gaia_429_failures_nohup.sh --stats-only
#   CONCURRENCY=3 RESULTS=results/gaia_bench_agent_20260522_004730.json ./scripts/rerun_gaia_429_failures_nohup.sh
#   STATS_ONLY=1 ./scripts/rerun_gaia_429_failures_nohup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

STATS_ONLY="${STATS_ONLY:-0}"
if [ "${1:-}" = "--stats-only" ]; then
  STATS_ONLY=1
fi

RESULTS="${RESULTS:-results/gaia_bench_agent_20260522_004730.json}"
GAIA_DATA="${GAIA_DATA:-data/gaia_validate_level1.jsonl}"
OUTPUT_DATA="${OUTPUT_DATA:-data/gaia_validate_level1_429_rerun.jsonl}"
CONCURRENCY="${CONCURRENCY:-3}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_gaia_429_rerun_${TIMESTAMP}.log}"

if [ ! -f "$RESULTS" ]; then
  echo "Error: results file not found: $RESULTS"
  exit 1
fi

if [ ! -f "$GAIA_DATA" ]; then
  echo "Error: GAIA data file not found: $GAIA_DATA"
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

PY_ARGS=(
  scripts/rerun_gaia_429_failures.py
  --results "$RESULTS"
  --gaia-data "$GAIA_DATA"
  --output-data "$OUTPUT_DATA"
  --results-dir "$RESULTS_DIR"
  --concurrency "$CONCURRENCY"
  --manager-tools "$MANAGER_TOOLS"
  --search-tools "$SEARCH_TOOLS"
)

if [ "$STATS_ONLY" = "1" ]; then
  echo "====================================================="
  echo "GAIA 429 / TPM 限流失败任务统计"
  echo "Results: $RESULTS"
  echo "GAIA data: $GAIA_DATA"
  echo "Output jsonl: $OUTPUT_DATA"
  echo "====================================================="
  "${RUNNER[@]}" "${PY_ARGS[@]}" --no-run
  exit 0
fi

echo "====================================================="
echo "Rerunning GAIA 429 / TPM failures (nohup)"
echo "Results: $RESULTS"
echo "GAIA data: $GAIA_DATA"
echo "Rerun jsonl: $OUTPUT_DATA"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL_NAME"
echo "Manager tools: $MANAGER_TOOLS"
echo "Search tools: $SEARCH_TOOLS"
echo "Log file: $LOG_FILE"
echo "====================================================="
echo "Monitor with: tail -f $LOG_FILE"

touch "$LOG_FILE"

nohup env PYTHONUNBUFFERED="$PYTHONUNBUFFERED" PYTHONIOENCODING="$PYTHONIOENCODING" "${RUNNER[@]}" \
  "${PY_ARGS[@]}" \
  --run \
  --log-file "$LOG_FILE" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started in background. PID: $PID"
echo "Done. Log: $LOG_FILE | Results: $RESULTS_DIR"
