#!/bin/bash
# Load project proxy + .env for shell scripts and nohup runs.

set -a
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/scripts/proxy.env.sh" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/scripts/proxy.env.sh"
fi
if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/.env"
fi
set +a
