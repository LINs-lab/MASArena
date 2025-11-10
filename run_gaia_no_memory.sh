#!/bin/bash
# Script to run Gaia benchmark with SmolAgents and MeLO Memory.

# --- Configuration ---
BENCHMARK="gaia"
AGENT_SYSTEM="smolagents"
DATA_FILE="data/gaia_validate.jsonl"
LIMIT=1
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

# Exit with the same status as the Python script
exit $?