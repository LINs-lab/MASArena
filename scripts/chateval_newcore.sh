#!/bin/bash
# 后台运行 ChatEval NewCore 测试，输出写入日志。
# 用法: ./scripts/chateval_newcore.sh [LOG_DIR]
# 例:   ./scripts/chateval_newcore.sh
#       ./scripts/chateval_newcore.sh logs/chateval_newcore

set -e

LOG_DIR=${1:-logs/chateval_newcore}
timestamp=$(date +%Y%m%d%H%M%S)
LOG_FILE="${LOG_DIR}/chateval_newcore_${timestamp}.log"

mkdir -p "$LOG_DIR"

echo "====================================================="
echo "Running: uv run tests/test_chateval_newcore.py"
echo "Log File: $LOG_FILE"
echo "====================================================="

nohup uv run tests/test_chateval_newcore.py > "$LOG_FILE" 2>&1 &

echo "Started in background. PID: $!"
echo "Monitor with: tail -f $LOG_FILE"

exit 0