#!/bin/bash
# 从 bench_agent GAIA 结果 JSON 中统计全部错题，生成 jsonl 并后台重跑。
#
# Usage:
#   ./scripts/rerun_gaia_failed_nohup.sh --stats-only
#   ./scripts/rerun_gaia_failed_nohup.sh
#   RESULTS=results/gaia_bench_agent_20260522_004736.json CONCURRENCY=3 ./scripts/rerun_gaia_failed_nohup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

STATS_ONLY="${STATS_ONLY:-0}"
if [ "${1:-}" = "--stats-only" ]; then
  STATS_ONLY=1
fi

RESULTS="${RESULTS:-results/gaia_bench_agent_20260522_004736.json}"
BASELINE="${BASELINE:-$RESULTS}"
CONCURRENCY="${CONCURRENCY:-3}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/bench_agent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_agent_gaia_failed_rerun_${TIMESTAMP}.log}"

if [ ! -f "$RESULTS" ]; then
  echo "Error: results file not found: $RESULTS"
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

STEM="$(basename "$RESULTS" .json)"
OUTPUT_DATA="${OUTPUT_DATA:-data/gaia_failed_rerun_${STEM}.jsonl}"

PY_ARGS=(
  scripts/rerun_gaia_429_failures.py
  --results "$RESULTS"
  --failed-only
  --output-data "$OUTPUT_DATA"
  --results-dir "$RESULTS_DIR"
  --concurrency "$CONCURRENCY"
  --manager-tools "$MANAGER_TOOLS"
  --search-tools "$SEARCH_TOOLS"
)

if [ "$STATS_ONLY" = "1" ]; then
  echo "====================================================="
  echo "GAIA 错题统计 + 生成重跑 jsonl"
  echo "Results: $RESULTS"
  echo "Output:  $OUTPUT_DATA"
  echo "====================================================="
  "${RUNNER[@]}" "${PY_ARGS[@]}" --no-run
  exit 0
fi

echo "====================================================="
echo "Rerunning GAIA failed problems (nohup)"
echo "Results: $RESULTS"
echo "Rerun jsonl: $OUTPUT_DATA"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL_NAME"
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
echo ""
echo "重跑完成后合并准确率:"
echo "  MERGE_ONLY=1 BASELINE=$RESULTS RERUN_RESULTS=results/gaia_bench_agent_<新时间戳>.json \\"
echo "    ./scripts/rerun_gaia_429_failures_nohup.sh --merge"
