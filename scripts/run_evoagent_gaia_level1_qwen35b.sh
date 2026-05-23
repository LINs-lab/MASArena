#!/bin/bash
# Run EvoAgent on the first 10 GAIA validation level-1 samples with Qwen3.5 35B.
#
# Usage:
#   ./scripts/run_evoagent_gaia_level1_qwen35b.sh
#   LIMIT=20 EVO_WORKER_DELAY_SECONDS=10 ./scripts/run_evoagent_gaia_level1_qwen35b.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f scripts/load_project_env.sh ] && . scripts/load_project_env.sh

BENCHMARK="${BENCHMARK:-gaia}"
AGENT_SYSTEM="${AGENT_SYSTEM:-evoagent}"
DATA_PATH="${DATA_PATH:-data/gaia_validate_level1.jsonl}"
LIMIT="${LIMIT:-2}"
CONCURRENCY="${CONCURRENCY:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/evoagent_gaia}"
MANAGER_TOOLS="${MANAGER_TOOLS:-ALL}"
SEARCH_TOOLS="${SEARCH_TOOLS:-ALL}"
PROBLEM_TIMEOUT_SECONDS="${PROBLEM_TIMEOUT_SECONDS:-3600}"
EVO_AGENT_TIMEOUT_SECONDS="${EVO_AGENT_TIMEOUT_SECONDS:-600}"
EVO_WORKER_DELAY_SECONDS="${EVO_WORKER_DELAY_SECONDS:-5}"
BROWSER_TIMEOUT_SECONDS="${BROWSER_TIMEOUT_SECONDS:-120}"
SEARCH_JINA_TIMEOUT="${SEARCH_JINA_TIMEOUT:-45}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/evoagent_gaia_level1_qwen35b_${TIMESTAMP}.log}"

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
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-900}"
export MAS_ARENA_BROWSER_TIMEOUT_SECONDS="$BROWSER_TIMEOUT_SECONDS"
export MAS_ARENA_EVO_AGENT_TIMEOUT_SECONDS="$EVO_AGENT_TIMEOUT_SECONDS"
export MAS_ARENA_EVO_WORKER_DELAY_SECONDS="$EVO_WORKER_DELAY_SECONDS"
export MAS_ARENA_EVO_STEP_TIMEOUT_SECONDS="${MAS_ARENA_EVO_STEP_TIMEOUT_SECONDS:-120}"
export MAS_ARENA_EVO_SUMMARY_TIMEOUT_SECONDS="${MAS_ARENA_EVO_SUMMARY_TIMEOUT_SECONDS:-600}"
export SEARCH_JINA_TIMEOUT
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "====================================================="
echo "Running EvoAgent GAIA validation level 1"
echo "Benchmark: $BENCHMARK"
echo "Agent system: $AGENT_SYSTEM"
echo "Data: $DATA_PATH"
echo "Limit: $LIMIT"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL_NAME"
echo "Manager tools: $MANAGER_TOOLS"
echo "Search tools: $SEARCH_TOOLS"
echo "Problem timeout seconds: $PROBLEM_TIMEOUT_SECONDS"
echo "Evo worker timeout seconds: $EVO_AGENT_TIMEOUT_SECONDS"
echo "Evo worker delay seconds: $EVO_WORKER_DELAY_SECONDS"
echo "Browser timeout seconds: $BROWSER_TIMEOUT_SECONDS"
echo "Jina timeout seconds: $SEARCH_JINA_TIMEOUT"
echo "OpenAI API timeout seconds: $OPENAI_API_TIMEOUT"
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
  --problem-timeout-seconds "$PROBLEM_TIMEOUT_SECONDS" \
  --evo-agent-timeout-seconds "$EVO_AGENT_TIMEOUT_SECONDS" \
  --evo-worker-delay-seconds "$EVO_WORKER_DELAY_SECONDS" \
  --async-run \
  --concurrency "$CONCURRENCY" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started in background. PID: $PID"
echo "Done. Log: $LOG_FILE | Results: $RESULTS_DIR"
