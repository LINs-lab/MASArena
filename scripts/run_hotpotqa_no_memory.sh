#!/bin/bash
# Script to run Aime benchmark with SmolAgents and without MeLO Memory.

# --- Configuration ---
BENCHMARK="hotpotqa"
AGENT_SYSTEM="smolagents"
DATA_FILE="data/hotpotqa_test.jsonl"
LIMIT=200
CONCURRENCY=20
# --- End Configuration ---

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."

fi

export MELO_DEBUG=1

# Print header
echo "====================================================="
echo "Running Custom Benchmark"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Data File: $DATA_FILE"
echo "Limit: $LIMIT"
echo "====================================================="

# Execute the command
.venv/bin/python main.py \
  --benchmark "$BENCHMARK" \
  --agent-system "$AGENT_SYSTEM" \
  --data "$DATA_FILE" \
  --limit "$LIMIT" \
  --async-run \
  --concurrency "$CONCURRENCY" \

# Exit with the same status as the Python script
exit $?
