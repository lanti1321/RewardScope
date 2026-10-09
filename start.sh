#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo "首次使用请先运行：bash setup.sh"
  exit 1
fi
exec .venv/bin/python -m rl_workbench.server "$@"
