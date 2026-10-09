#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
echo "安装完成。运行 bash start.sh，再打开 http://127.0.0.1:8765"
