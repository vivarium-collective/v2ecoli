"""Regression test for issue #340.

`reports/multigeneration_report.py` used to reconstruct its per-generation mass
time-series by reading ``emitter_instance.history``. No supported emitter keeps
every timestep in memory — ``ParquetEmitter`` (the default) streams each tick to
disk — so that access raised
``AttributeError: 'ParquetEmitter' object has no attribute 'history'`` right
after an otherwise-successful run. The fix accumulates the mass series from live
composite state per chunk instead, which works for every emitter type.

This test loads the report module (a top-level script, not a package) by path
and drives ``_run_generation`` with a fake composite whose emitter deliberately
has no ``.history`` attribute, mirroring ``ParquetEmitter``. It needs no ParCa
cache and runs in milliseconds.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports",
    "multigeneration_report.py",
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "multigeneration_report", _REPORT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ParquetLikeEmitter:
    """Stand-in for ParquetEmitter: streams to disk, exposes no ``.history``."""

    out_uri = "/tmp/parquet-runs/fake"


class _FakeComposite:
    """Minimal composite: grows dry_mass per chunk, then 'divides' by detaching
    agent 0 (raising a division-flavoured error, exactly like the real run)."""

    def __init__(self, divide_at: float = 150.0) -> None:
        self._t = 0.0
        self._divide_at = divide_at
        self.state = {
            "agents": {
                "0": {
                    "listeners": {
                        "mass": {
                            "dry_mass": 300.0,
                            "cell_mass": 900.0,
                            "protein_mass": 500.0,
                            "dna_mass": 30.0,
                            "rRna_mass": 200.0,
                            "tRna_mass": 40.0,
                            "mRna_mass": 20.0,
                            "smallMolecule_mass": 100.0,
                        }
                    },
                    "emitter": {"instance": _ParquetLikeEmitter()},
                }
            }
        }

    def run(self, chunk: float) -> None:
        self._t += chunk
        cell = self.state["agents"].get("0")
        if cell is None:
            return
        cell["listeners"]["mass"]["dry_mass"] += 100.0
        if self._t >= self._divide_at:
            del self.state["agents"]["0"]
            raise Exception("cell _remove: division")


def test_run_generation_reconstructs_without_emitter_history():
    """The exact #340 crash: a ParquetEmitter-like emitter (no .history) must
    not raise, and a mass time-series must be reconstructed from live state."""
    mod = _load_report_module()

    comp = _FakeComposite(divide_at=150.0)
    result = mod._run_generation(comp, gen_idx=0, max_duration=300.0)

    assert result.divided is True
    # Two SNAPSHOT_INTERVAL chunks captured before division (t=50, t=100 grew
    # mass; t=150 detaches the agent so no post-division snapshot is kept).
    assert result.snapshots, "no snapshots reconstructed from live state"
    assert all(s["dry_mass"] > 0 for s in result.snapshots)
    # Downstream visualization shape: each snapshot carries the mass fields.
    for field_name in ("time", "dry_mass", "cell_mass", "protein_mass"):
        assert field_name in result.snapshots[0]
    # Monotonic growth reconstructed correctly.
    assert result.snapshots[0]["dry_mass"] == pytest.approx(400.0)
    assert result.final_dry_mass == pytest.approx(result.snapshots[-1]["dry_mass"])


def test_report_module_has_no_emitter_history_access():
    """Guard against the pattern regressing: no executable line may read
    ``.history`` off an emitter instance (comments referencing the old bug are
    fine)."""
    with open(_REPORT_PATH, encoding="utf-8") as fh:
        code_lines = [
            line for line in fh if not line.lstrip().startswith("#")
        ]
    assert not any("emitter_instance.history" in line for line in code_lines)
