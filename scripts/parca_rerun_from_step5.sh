#!/usr/bin/env bash
# Rerun steps 6-9 of the ParCa pipeline using the cached step-5 checkpoint.
# Turns a 70-minute fresh run into ~15 seconds.
#
# Requires out/sim_data/checkpoint_step_5.pkl from a prior run.  The
# resume-path bug that used to crash step 6 was fixed in composite.py
# (port values are now seeded at their nested STORE_PATH locations).
#
# Usage:  scripts/rerun_from_step5.sh [extra args forwarded to parca_bigraph.py]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${REPO_ROOT}/out/sim_data/checkpoint_step_5.pkl"
PY="${PYTHON:-/Users/eranagmon/code/vivarium-ecoli/venv/bin/python}"

if [[ ! -f "$CKPT" ]]; then
  echo "Error: $CKPT not found.  Run scripts/parca_bigraph.py end-to-end first." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" "${REPO_ROOT}/scripts/parca_bigraph.py" \
  --mode fast --cpus 4 \
  --resume-from-step 6 --resume-pickle "$CKPT" \
  -o "${REPO_ROOT}/out/sim_data" \
  "$@"
