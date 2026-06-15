"""Shared helper: derive the canonical ``run_id`` for a run directory so that
rendered figures can carry it in their ``.meta.json``.

The canonical run id is the run's ``experiment_id`` — the top parquet
partition key (``.../configuration/experiment_id=<id>/...``). Figures that
carry it can be traced back to the exact config that produced the run via
``python -m pbg_superpowers.provenance <figure>`` (see
docs/conventions/run-provenance.md).

Resolution:
  1. Find an ``experiment_id=<id>`` partition segment anywhere under the run
     dir (authoritative — this is literally the partition key the emitter
     wrote).
  2. Fall back to the run dir's basename (the runner names
     ``out/<experiment_id>/<experiment_id>/`` so the basename is the id).
"""
from __future__ import annotations

import glob
import os


def run_id_from_run_dir(run_dir: str) -> str:
    """Return the canonical run_id (experiment_id) for ``run_dir``."""
    run_dir = run_dir.rstrip("/")
    # 1) authoritative: experiment_id=<id> partition segment.
    for cfg in glob.glob(os.path.join(run_dir, "**", "experiment_id=*"),
                         recursive=True):
        seg = os.path.basename(cfg)
        if seg.startswith("experiment_id="):
            return seg.split("=", 1)[1]
    # 2) fallback: directory basename.
    return os.path.basename(run_dir)
