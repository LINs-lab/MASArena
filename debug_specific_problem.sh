#!/bin/bash
# Script to debug a specific problem ID in Gaia benchmark with pass@k metric.
# This script is specifically for debugging problem ID: c61d22de-5f6c-4958-a7f6-5e9707bd3466

# --- Configuration ---
BENCHMARK="gaia"
AGENT_SYSTEM="smolagents"
DATA_FILE="data/gaia_validate_level1.jsonl"  # 默认使用level1数据
PROBLEM_ID="c61d22de-5f6c-4958-a7f6-5e9707bd3466"
PASS_AT_K=2
timestamp=$(date +%Y%m%d%H%M%S)
LOG_DIR="logs/gaia_debug"
LOG_FILE="${LOG_DIR}/gaia_debug_${PROBLEM_ID}_${timestamp}.log"
# --- End Configuration ---

mkdir -p ${LOG_DIR}
export MODEL_NAME="gpt-4.1"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
  echo "Using model: $MODEL_NAME"
fi

# Print header
echo "====================================================="
echo "Debugging Gaia Benchmark for Specific Problem"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Problem ID: $PROBLEM_ID"
echo "Pass@K: $PASS_AT_K"
echo "Log File: $LOG_FILE"
echo "====================================================="

# Execute the command in the foreground for easier debugging
python main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_FILE" \
  --problem-id "$PROBLEM_ID" \
  --pass_at_k "$PASS_AT_K" \
  --async-run \
  --concurrency 1 | tee "$LOG_FILE"

echo "Debug completed. Log saved to: $LOG_FILE"






