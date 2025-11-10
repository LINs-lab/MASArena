#!/bin/bash
# Script to run Gaia benchmark with pass@k metric.

# --- Configuration ---
BENCHMARK="gaia"
AGENT_SYSTEM="smolagents"
DATA_FILE="data/gaia_validate_level1.jsonl"
LIMIT=1
PASS_AT_K=2
timestamp=$(date +%Y%m%d%H%M%S)
LOG_FILE="logs/gaia_pass_at_k/gaia_pass_at_k_${timestamp}.log"
# --- End Configuration ---

mkdir -p logs/gaia_pass_at_k
export MODEL_NAME="gpt-4.1"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
  echo "Using model: $MODEL_NAME"
fi

# Print header
echo "====================================================="
echo "Running Gaia Benchmark with Pass@K"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Data File: $DATA_FILE"
echo "Limit: $LIMIT"
echo "Pass@K: $PASS_AT_K"
echo "Log File: $LOG_FILE"
echo "====================================================="

# Execute the command in the background
nohup python main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_FILE" \
  --limit "$LIMIT" \
  --pass_at_k "$PASS_AT_K" \
  --async-run \
  --concurrency 10 > "$LOG_FILE" 2>&1 &

echo "Script started in the background. PID: $!"
echo "You can monitor the output with: tail -f $LOG_FILE"

# Exit immediately
exit 0
