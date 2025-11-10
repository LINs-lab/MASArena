#!/bin/bash
# Baseline test script: qwen3-32b, no memory, no RL

BENCHMARK="gaia"
AGENT_SYSTEM="smolagents"
DATA_FILE="data/gaia_validate_level1.jsonl"
LIMIT=1
CONCURRENCY=10
# MEMORY="melo"
# Create logs directory
mkdir -p logs

# Activate virtual environment if exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
fi

# Set model for smolagents
export MODEL_NAME="qwen3-32b"
export OPENAI_API_TIMEOUT="600" # Increase timeout to 5 minutes
echo "Using model: $MODEL_NAME"

echo "====================================================="
echo "Baseline test: $AGENT_SYSTEM, no memory, no RL"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Data File: $DATA_FILE"
echo "Limit: $LIMIT"
# echo "Memory: $MEMORY"
echo "====================================================="

# Run in background and redirect logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${MODEL_NAME}_${AGENT_SYSTEM}_baseline_${TIMESTAMP}.log"
nohup python main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_FILE" \
  --limit "$LIMIT" \
  --async-run \
  --concurrency "$CONCURRENCY" \
  --verbose > "$LOG_FILE" 2>&1 &
echo "Started baseline test in background. Logs at $LOG_FILE"
