"""Golden-trajectory parity gate for the polypeptide-elongation refactor.

The default-wired elongation variant (SteadyState) must reproduce the
baseline trajectory bit-for-bit. Regenerate the golden ONLY intentionally:
    V2_WRITE_GOLDEN=1 .venv/bin/pytest tests/test_polypeptide_elongation_parity.py
"""
import json
import os

import numpy as np
import pytest

CACHE = "out/cache"
GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                      "polypeptide_elongation_baseline.json")
STEPS = 100  # short, CI-friendly; parity drift shows within a few ticks

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


def _trajectory():
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import fg_magnitude
    c = build_composite("baseline", cache_dir=CACHE, seed=0)
    a = c.state["agents"]["0"]
    rec = []
    for _ in range(STEPS):
        c.run(1)
        mass = a["listeners"]["mass"]
        rec.append(round(float(fg_magnitude(mass["dry_mass"])), 6))
    bulk = a.get("bulk")
    bulk_sum = int(np.nansum(bulk["count"])) if getattr(bulk, "dtype", None) and bulk.dtype.names else int(np.nansum(bulk))
    return {"dry_mass": rec, "bulk_total_at_end": bulk_sum}


def test_baseline_elongation_trajectory_matches_golden():
    traj = _trajectory()
    if os.environ.get("V2_WRITE_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(traj, f, indent=1)
        pytest.skip("wrote golden")
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert traj["dry_mass"] == golden["dry_mass"], (
        "dry_mass trajectory drifted from golden — elongation refactor changed behaviour")
    assert traj["bulk_total_at_end"] == golden["bulk_total_at_end"]
