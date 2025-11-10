#!/bin/bash
# Script to run Aime benchmark with SmolAgents and MeLO Memory.

# --- Configuration ---
BENCHMARK="aime"
AGENT_SYSTEM="simple_smolagents"
DATA_FILE="data/aime_test.jsonl"
LIMIT=30
CONCURRENCY=5
MEMORY_TYPE="melo_memory"
# --- End Configuration ---

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
fi

# Print header
echo "====================================================="
echo "Running Custom Benchmark"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Data File: $DATA_FILE"
echo "Limit: $LIMIT"
echo "Memory Type: $MEMORY_TYPE"
echo "====================================================="

# Execute the command
python main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_FILE" \
  --limit "$LIMIT" \
  --memory-type "$MEMORY_TYPE" \
  --async-run \
  --concurrency "$CONCURRENCY" \


# Exit with the same status as the Python script
exit $?
