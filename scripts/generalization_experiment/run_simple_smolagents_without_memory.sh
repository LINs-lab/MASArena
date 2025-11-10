#!/bin/bash
# Script to run several benchmarks with SmolAgents and without MeLO Memory.

# Benchmarks to run (these names are used for both --benchmark and to build the
# data filename data/<benchmark>_test_200.jsonl)
BENCHMARKS=("aime" "bbh" "math")
AGENT_SYSTEM="simple_smolagents"
LIMIT=200
CONCURRENCY=20
RUNS=5

# Logs directory
LOG_DIR=".log/without_memory"
mkdir -p "$LOG_DIR"

# Set debug env if desired
export MELO_DEBUG=1

# A single timestamp for the whole script run (ISO 8601, UTC)
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Experiment timestamp: $TS"

# Activate virtual environment if it exists (kept inside loop in case it changes)
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
fi

echo "Starting runs: benchmarks=${BENCHMARKS[*]}, runs_per_benchmark=$RUNS, limit=$LIMIT, concurrency=$CONCURRENCY"

for BENCHMARK in "${BENCHMARKS[@]}"; do
  DATA_FILE="data/${BENCHMARK}_test_200.jsonl"
  if [ ! -f "$DATA_FILE" ]; then
    echo "Warning: data file not found for $BENCHMARK: $DATA_FILE" >&2
    # continue to next benchmark
    continue
  fi

  for i in $(seq 1 $RUNS); do
    OUT_FILE="$LOG_DIR/${TS}_${BENCHMARK}_run_${i}.out"
    ERR_FILE="$LOG_DIR/${TS}_${BENCHMARK}_run_${i}.err"
    RC_FILE="$LOG_DIR/${TS}_${BENCHMARK}_run_${i}.rc"

    echo "-----------------------------------------------------"
    echo "Running benchmark=$BENCHMARK run=$i"
    echo "Data: $DATA_FILE"
    echo "Stdout -> $OUT_FILE"
    echo "Stderr -> $ERR_FILE"

    # Run the benchmark and capture stdout/stderr separately
    .venv/bin/python main.py \
      --benchmark "$BENCHMARK" \
      --agent-system "$AGENT_SYSTEM" \
      --data "$DATA_FILE" \
      --limit "$LIMIT" \
      --async-run \
      --concurrency "$CONCURRENCY" \
      >"$OUT_FILE" 2>"$ERR_FILE"

    rc=$?
    echo $rc > "$RC_FILE"
    echo "Run finished: benchmark=$BENCHMARK run=$i rc=$rc"
  done
done

echo "All runs completed. Logs are under $LOG_DIR"
exit 0
