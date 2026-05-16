"""Demo: feed the evaluator a synthetic history and verify the
expected_behavior grammar end-to-end.

No runs.db required. Useful so the user can:
  pytest studies/dnaa-01-expression-dynamics/tests/test_with_synthetic_history.py -v

and see the assertion library work today, while waiting for the first
real baseline run.

The synthetic history mimics a plausible steady-state DnaA dynamics for
the BASELINE run, and a translation-knockout exponential decay for the
STOP-SYNTHESIS variant.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
import yaml

from _behaviors import DNAA_MONOMER_ID, evaluate

STUDY_DIR = Path(__file__).resolve().parents[1]


def _make_state(dnaA_count: int, *, n_bound: int = 0, n_init: int = 0) -> dict:
    """Build one snapshot dict in the shape conftest.bulk_count expects."""
    return {
        "agents": {
            "0": {
                "bulk": {
                    "id": [DNAA_MONOMER_ID, "EG10235_RNA"],
                    "count": [dnaA_count, 12],
                },
                "listeners": {
                    "rna_synth_prob": {
                        "n_actual_bound": [n_bound],
                    },
                    "rnap_data": {
                        "rna_init_event": [n_init],
                    },
                },
            }
        }
    }


@pytest.fixture
def synthetic_baseline_history() -> list[dict]:
    """40 snapshots, DnaA hovering at ~520 with small noise + a coarse
    inverse correlation between n_bound and n_init (autorepression)."""
    rng = random.Random(0)
    out = []
    for step in range(40):
        # mean 520, sd 25 -> CV ~0.05 (well under the 10% threshold)
        count = int(rng.gauss(520, 25))
        # autorepression: when n_bound is high, n_init is low.
        n_bound = 80 + (step % 10)         # walks 80..89
        n_init = 12 - (n_bound - 80) // 2  # inversely correlated with n_bound
        out.append({"step": step, "time": step * 60.0, "state": _make_state(count, n_bound=n_bound, n_init=n_init)})
    return out


@pytest.fixture
def synthetic_stop_synthesis_history() -> list[dict]:
    """Exponential decay from 520 with half-life ~1 doubling time (40 min)."""
    out = []
    initial = 520
    half_life_steps = 40   # 40 steps of 1 minute = one doubling time
    for step in range(60):
        count = int(initial * math.exp(-step * math.log(2) / half_life_steps))
        out.append({"step": step, "time": step * 60.0, "state": _make_state(count)})
    return out


def _entries():
    spec = yaml.safe_load((STUDY_DIR / "study.yaml").read_text())
    return spec.get("expected_behavior") or []


_BASELINE_ENTRIES = [e for e in _entries()
                     if (e.get("given") or {}).get("run") == "baseline"
                     and e.get("status") == "implemented"]

_STOP_ENTRIES = [e for e in _entries()
                 if (e.get("given") or {}).get("run") == "variant"
                 and (e.get("given") or {}).get("variant") == "stop-dnaA-synthesis"
                 and e.get("status") == "implemented"]


@pytest.mark.parametrize("entry", _BASELINE_ENTRIES,
                          ids=[e["name"] for e in _BASELINE_ENTRIES])
def test_baseline_behaviors_against_synthetic(entry, synthetic_baseline_history):
    passed, message = evaluate(entry, synthetic_baseline_history)
    assert passed, f"{entry['en']}\n  -> {message}"


@pytest.mark.parametrize("entry", _STOP_ENTRIES,
                          ids=[e["name"] for e in _STOP_ENTRIES])
def test_stop_synthesis_behaviors_against_synthetic(entry, synthetic_stop_synthesis_history):
    passed, message = evaluate(entry, synthetic_stop_synthesis_history)
    assert passed, f"{entry['en']}\n  -> {message}"
