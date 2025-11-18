#!/bin/bash
# run_aflow_optimize.sh - Run AFlow optimization pipeline with configurable parameters

# Usage:
#   ./run_aflow_optimize.sh [benchmark] [graph_path] [optimized_path] [validation_rounds] [eval_rounds] [max_rounds]

BENCHMARK=${1:-humaneval}
GRAPH_PATH=${2:-"mas_arena/configs/aflow"}
OPTIMIZED_PATH=${3:-"example/aflow/humaneval/optimization"}
VALIDATION_ROUNDS=${4:-1}
EVAL_ROUNDS=${5:-1}
MAX_ROUNDS=${6:-3}
TRAIN_SIZE=${7:-40}
TEST_SIZE=${8:-20}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using system environment variables."
fi

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo ""
echo "====================================================="
echo "🚀 Running AFlow Optimization Pipeline"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Graph Path: $GRAPH_PATH"
echo "Optimized Path: $OPTIMIZED_PATH"
echo "Validation Rounds: $VALIDATION_ROUNDS"
echo "Evaluation Rounds: $EVAL_ROUNDS"
echo "Max Optimization Rounds: $MAX_ROUNDS"
echo "Size of the training set for evaluation: $TRAIN_SIZE"
echo "Size of the training set for evaluation: $TEST_SIZE"
echo "====================================================="
echo ""

python example/aflow/run_aflow_optimize.py \
    --benchmark "$BENCHMARK" \
    --graph_path "$GRAPH_PATH" \
    --optimized_path "$OPTIMIZED_PATH" \
    --validation_rounds "$VALIDATION_ROUNDS" \
    --eval_rounds "$EVAL_ROUNDS" \
    --max_rounds "$MAX_ROUNDS" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE"

exit $?