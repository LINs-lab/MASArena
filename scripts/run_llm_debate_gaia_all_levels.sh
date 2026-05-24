#!/bin/bash
# Run LLM Debate on GAIA level1 → level2 → level3 sequentially.
# Each level starts only after the previous one finishes.
#
# Usage:
#   nohup ./scripts/run_llm_debate_gaia_all_levels.sh > logs/llm_debate_gaia/all_levels_<ts>.log 2>&1 &
#   MODEL_NAME=gpt-4.1 CONCURRENCY=10 ./scripts/run_llm_debate_gaia_all_levels.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

BENCHMARK="${BENCHMARK:-gaia}"
AGENT_SYSTEM="${AGENT_SYSTEM:-llm_debate}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
CONCURRENCY="${CONCURRENCY:-20}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/llm_debate_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [ -x ".venv/bin/python" ]; then
  RUNNER=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  RUNNER=(python3)
fi

export MODEL_NAME
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-600}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

run_level() {
  local level="$1"
  local data_path="$2"
  local log_file="$LOG_DIR/llm_debate_gaia_level${level}_${MODEL_NAME}_${TIMESTAMP}.log"

  echo ""
  echo "============================================================"
  echo "  Level ${level} — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  Data:        $data_path"
  echo "  Concurrency: $CONCURRENCY"
  echo "  Model:       $MODEL_NAME"
  echo "  Log:         $log_file"
  echo "============================================================"

  "${RUNNER[@]}" main.py \
    --benchmark "$BENCHMARK" \
    --agent-system "$AGENT_SYSTEM" \
    --data "$data_path" \
    --results-dir "$RESULTS_DIR" \
    --model-name "$MODEL_NAME" \
    --manager-tools "$MANAGER_TOOLS" \
    --search-tools "$SEARCH_TOOLS" \
    --log-file "$log_file" \
    --async-run \
    --concurrency "$CONCURRENCY" \
    2>&1 | tee "$log_file"

  echo ""
  echo "  Level ${level} finished — $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "====================================================="
echo "LLM Debate GAIA all levels"
echo "Model: $MODEL_NAME  |  Concurrency: $CONCURRENCY"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================="

run_level 1 "data/gaia_validate_level1.jsonl"
run_level 2 "data/gaia_validate_level2.jsonl"
run_level 3 "data/gaia_validate_level3.jsonl"

echo ""
echo "====================================================="
echo "All levels done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $RESULTS_DIR"
echo "====================================================="
