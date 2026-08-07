"""Shared helper: derive the canonical ``run_id`` for a run directory so that
rendered figures can carry it in their ``.meta.json``.

The implementation moved to ``v2ecoli.library.run_provenance`` so library code
(the sim-vector cache) can use it too; this module re-exports it for the render
scripts that already import from here.

The canonical run id is the run's ``experiment_id`` — the top parquet
partition key (``.../configuration/experiment_id=<id>/...``). Figures that
carry it can be traced back to the exact config that produced the run via
``python -m viva_superpowers.provenance <figure>`` (see
docs/conventions/run-provenance.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.library.run_provenance import run_id_from_run_dir  # noqa: E402,F401

__all__ = ["run_id_from_run_dir"]
