#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
microduck_commit=cb70b792312d559a4da09064d92009079671815f
if ! command -v uv >/dev/null 2>&1; then
  echo "请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
if [[ ! -d upstream/microduck_rl ]]; then
  mkdir -p upstream
  git clone https://github.com/pollen-robotics/microduck_rl.git upstream/microduck_rl
  git -C upstream/microduck_rl checkout "$microduck_commit"
fi
if [[ "$(git -C upstream/microduck_rl rev-parse HEAD)" != "$microduck_commit" ]]; then
  echo "上游版本与本适配器验证的提交不一致，请使用提交 $microduck_commit 的独立副本。"
  exit 1
fi
# The pinned upstream commit has manifest/lock drift. Preserve the supplied
# lockfile instead of resolving/upgrading it in place.
UV_HTTP_TIMEOUT=600 uv sync --frozen --project upstream/microduck_rl
mkdir -p .runtime
upstream/microduck_rl/.venv/bin/python rl_workbench/microduck_bridge.py catalog .runtime/microduck_catalog.json
echo "官方 MicroDuck 环境已就绪。运行 bash start.sh，打开 /microduck。"
