#!/bin/bash
# Script to run Gaia benchmark with Jarvis (no memory) and pass@k metric for level 1.
# Usage: ./scripts/run_gaia_level1_jarvis_no_memory.sh [PASS_AT_K]
# Example: ./scripts/run_gaia_level1_jarvis_no_memory.sh 1

# --- Configuration ---
PASS_AT_K=${1:-1} # Default to 1 if not provided
# --- End Configuration ---

export MODEL_NAME="gpt-4.1" # Ensure this is set correctly

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
  echo "Using model: $MODEL_NAME"
fi

for LEVEL in 1; do
  # --- Level-specific Configuration ---
  BENCHMARK="gaia"
  AGENT_SYSTEM="jarvis"
  DATA_FILE="data/gaia_validate_level${LEVEL}.jsonl"
  LIMIT=0 # A limit of 0 means no limit
  timestamp=$(date +%Y%m%d%H%M%S)
  LOG_FILE="logs/gaia_jarvis_no_memory/level${LEVEL}/gaia_jarvis_level${LEVEL}_${timestamp}.log"
  # --- End Level-specific Configuration ---

  # Create log directory if it doesn't exist
  LOG_DIR=$(dirname "$LOG_FILE")
  mkdir -p "$LOG_DIR"

  # Print header
  echo "====================================================="
  echo "Running Gaia Benchmark with Pass@K for Level $LEVEL (Jarvis, No Memory)"
  echo "====================================================="
  echo "Benchmark: $BENCHMARK"
  echo "Agent System: $AGENT_SYSTEM"
  echo "Memory Type: No Memory"
  echo "LLM: $MODEL_NAME"
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

  echo "Script for level $LEVEL started in the background. PID: $!"
  echo "You can monitor the output with: tail -f $LOG_FILE"
  echo ""
done

# Exit immediately
exit 0
