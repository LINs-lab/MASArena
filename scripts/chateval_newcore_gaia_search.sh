#!/bin/bash
# 用 chateval_newcore 跑 GAIA 搜索题（使用已筛选好的 data/gaia_validate_level1_search_only.jsonl）
#
# 用法:
#   ./scripts/chateval_newcore_gaia_search.sh              # 跑全部搜索题，并发 5
#   ./scripts/chateval_newcore_gaia_search.sh 5             # 只跑前 5 题
#   ./scripts/chateval_newcore_gaia_search.sh 0 3           # 全部题，并发 3
#   LIMIT=5 ./scripts/chateval_newcore_gaia_search.sh       # 同上，只跑 5 题

set -e
cd "$(dirname "$0")/.."

DATA_PATH="${DATA_PATH:-data/gaia_validate_level1_search_only.jsonl}"
LIMIT="${LIMIT:-$1}"
CONCURRENCY="${CONCURRENCY:-${2:-5}}"
LOG_DIR="${LOG_DIR:-logs}"
ts=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-$LOG_DIR/chateval_newcore_gaia_search_${ts}.log}"

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: $DATA_PATH not found. Run: uv run python scripts/run_chateval_newcore_gaia_search_only.py --no-run"
  exit 1
fi

mkdir -p "$LOG_DIR"
echo "Data: $DATA_PATH | Log: $LOG_FILE | Limit: ${LIMIT:-all} | Concurrency: $CONCURRENCY"
echo "Running chateval_newcore on GAIA search-only problems (nohup)..."
echo "Monitor: tail -f $LOG_FILE"

nohup uv run python scripts/run_chateval_newcore_gaia_search_only.py \
  --skip-filter \
  --data_path "$DATA_PATH" \
  --output_data "$DATA_PATH" \
  --limit "${LIMIT:-0}" \
  --concurrency "$CONCURRENCY" \
  --log_file "$LOG_FILE" >> "$LOG_FILE" 2>&1 &

echo "Started in background. PID: $!"
echo "Done. Log: $LOG_FILE | Results: results/"
