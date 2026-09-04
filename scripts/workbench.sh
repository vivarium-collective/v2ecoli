#!/usr/bin/env bash
# Start/restart the vivarium-workbench dashboard on a STABLE port.
#
# Why this exists: `vivarium-workbench serve` (and `server restart`, which just
# calls it) does `args.port or _pick_free_port()` and never reads the previous
# port back from .pbg/server/server-info. So every restart hands you a new URL
# and invalidates the bookmark. Passing --port fixes it; this script makes sure
# we always pass the same one.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${WORKBENCH_PORT:-8322}"

if [ -f .pbg/server/server-info ]; then
  OLD=$(python3 -c "import json;print(json.load(open('.pbg/server/server-info')).get('pid',''))" 2>/dev/null || true)
  [ -n "${OLD:-}" ] && kill "$OLD" 2>/dev/null || true
  sleep 2
fi

.venv/bin/python3 -m vivarium_workbench.cli serve \
  --workspace "$PWD" --port "$PORT" --detach
echo "Dashboard pinned at http://127.0.0.1:${PORT}/#studies"
