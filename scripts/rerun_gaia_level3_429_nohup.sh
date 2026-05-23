#!/bin/bash
# 重跑 GAIA Level 3 中 429 / TPM 限流失败的题目（默认 14 题，来源 004736）。
#
# Usage:
#   ./scripts/rerun_gaia_level3_429_nohup.sh --stats-only
#   ./scripts/rerun_gaia_level3_429_nohup.sh

set -euo pipefail

export RESULTS="${RESULTS:-results/gaia_bench_agent_20260522_004736.json}"
export GAIA_DATA="${GAIA_DATA:-data/gaia_validate_level3.jsonl}"
export OUTPUT_DATA="${OUTPUT_DATA:-data/gaia_level3_429_rerun.jsonl}"
export CONCURRENCY="${CONCURRENCY:-3}"
export LEVEL=3

exec "$(dirname "$0")/rerun_gaia_429_failures_nohup.sh" "$@"
