#!/usr/bin/env bash
# 一键检查 .env 中所有 Jina API Key 余额
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/load_project_env.sh"
cd "$ROOT"
exec uv run python scripts/check_jina_balance.py "$@"
