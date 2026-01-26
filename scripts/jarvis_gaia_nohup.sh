#!/bin/bash
# 轻量版：把 `uv run tests/test_jarvis_gaia.py` 用 nohup 挂后台。
# 用法:
#   ./scripts/jarvis_gaia_nohup.sh [LOG_DIR]
# 例子:
#   ./scripts/jarvis_gaia_nohup.sh
#   ./scripts/jarvis_gaia_nohup.sh logs/jarvis_gaia

set -e

LOG_DIR=${1:-logs/jarvis_gaia_nohup}
timestamp=$(date +%Y%m%d%H%M%S)
LOG_FILE="${LOG_DIR}/jarvis_gaia_${timestamp}.log"

mkdir -p "$LOG_DIR"

echo "====================================================="
echo "Running: uv run tests/test_jarvis_gaia.py"
echo "Log File: $LOG_FILE"
echo "====================================================="

nohup uv run tests/test_jarvis_gaia.py > "$LOG_FILE" 2>&1 &

echo "Started in background. PID: $!"
echo "Monitor with: tail -f $LOG_FILE"

exit 0


