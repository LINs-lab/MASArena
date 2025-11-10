#!/bin/bash
# Script to run GAIA benchmark for levels 1, 2, and 3

BENCHMARK="gaia"
AGENT_SYSTEM="smolagents"
LIMIT=0 # 0 means no limit, will run all problems
CONCURRENCY=10
MEMORY_TYPE="melo_memory"

# Create logs directory
mkdir -p logs

# Activate virtual environment if exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
fi

# Set model for smolagents
export MODEL_NAME="qwen3-32b"
export OPENAI_API_TIMEOUT="600" # Increase timeout to 10 minutes
echo "Using model: $MODEL_NAME"

for LEVEL in 1 2 3; do
  DATA_FILE="data/gaia_validate_level${LEVEL}.jsonl"
  
  if [ ! -f "$DATA_FILE" ]; then
    echo "Data file not found for level ${LEVEL}: $DATA_FILE"
    continue
  fi

  echo "====================================================="
  echo "Running GAIA Level ${LEVEL}"
  echo "====================================================="
  echo "Benchmark: $BENCHMARK"
  echo "Agent System: $AGENT_SYSTEM"
  echo "Data File: $DATA_FILE"
  echo "Limit: $LIMIT (0 = no limit)"
  echo "Concurrency: $CONCURRENCY"
  echo "Memory Type: $MEMORY_TYPE"
  echo "====================================================="

  # Create level-specific log directory
  mkdir -p "logs/level${LEVEL}"

  # Run in background and redirect logs
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="logs/level${LEVEL}/${MODEL_NAME}_${AGENT_SYSTEM}_memory_test_level${LEVEL}_${TIMESTAMP}.log"
  nohup python main.py \
    --benchmark "$BENCHMARK" \
    --agent-system "$AGENT_SYSTEM" \
    --data "$DATA_FILE" \
    --limit "$LIMIT" \
    --async-run \
    --concurrency "$CONCURRENCY" \
    --memory-type "$MEMORY_TYPE" \
    --verbose > "$LOG_FILE" 2>&1 &
  echo "Started memory test for Level ${LEVEL} in background. Logs at $LOG_FILE"
  echo "====================================================="
  
  # Wait for a moment before starting the next level to avoid overwhelming the system
  sleep 5
done

echo "All baseline tests have been started in the background."
