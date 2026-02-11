#!/bin/bash
# 把 `uv run tests/test_chateval_newcore_gaia.py` 用 nohup 挂后台（GAIA level1）。
# 用法:
#   ./scripts/chateval_newcore_gaia_nohup.sh
#   ./scripts/chateval_newcore_gaia_nohup.sh logs/chateval_newcore_gaia

set -e

LOG_DIR=${1:-logs/chateval_newcore_gaia_nohup}
timestamp=$(date +%Y%m%d%H%M%S)
LOG_FILE="${LOG_DIR}/chateval_newcore_gaia_${timestamp}.log"

mkdir -p "$LOG_DIR"

echo "====================================================="
echo "Running: uv run tests/test_chateval_newcore_gaia.py"
echo "Log File: $LOG_FILE"
echo "====================================================="

nohup uv run tests/test_chateval_gaia.py > "$LOG_FILE" 2>&1 &

echo "Started in background. PID: $!"
echo "Monitor with: tail -f $LOG_FILE"

exit 0