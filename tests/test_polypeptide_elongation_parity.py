"""Golden-trajectory parity gate for the polypeptide-elongation refactor.

The default-wired SteadyState elongation must reproduce the baseline dry_mass
trajectory. The golden is generated from a developer's `out/cache`; CI rebuilds
the ParCa cache from scratch, so the trajectory differs by *float noise*
(~1e-5 fg from a different machine / float ordering) even with byte-identical
model code — locally it is bit-for-bit. We therefore compare with a TOLERANCE
that absorbs that cross-environment noise while still catching real behavioral
drift (a refactor bug diverges by >> 1e-3 fg, growing each tick). Regenerate
the golden ONLY intentionally:
    V2_WRITE_GOLDEN=1 .venv/bin/pytest tests/test_polypeptide_elongation_parity.py
"""
import json
import os

import numpy as np
import pytest

CACHE = "out/cache"
GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                      "polypeptide_elongation_baseline.json")
STEPS = 20  # any drift from a verbatim-move refactor shows within a few ticks;
            # the composite build dominates, so 20 ticks keeps this test cheap

# Builds + runs the baseline → a `sim` test (CI behavior job has the cache).
# The tolerant comparison below makes it CI-portable, so it gates on every PR.
pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE) and not os.environ.get("CI"),
        reason=f"cache dir {CACHE!r} not present",
    ),
]


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


# Hard signal-based timeout: under CI memory pressure build_composite can
# thrash, and pytest-timeout's default *thread* method can't interrupt a hang
# in native code — a single stuck test wedged the whole behavior job for ~43
# min once. SIGALRM kills it in minutes so a hang fails fast instead.
@pytest.mark.timeout(360, method="signal")
def test_baseline_elongation_trajectory_matches_golden():
    traj = _trajectory()
    if os.environ.get("V2_WRITE_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(traj, f, indent=1)
        pytest.skip("wrote golden")
    with open(GOLDEN) as f:
        golden = json.load(f)
    dm = np.asarray(traj["dry_mass"], dtype=float)
    gm = np.asarray(golden["dry_mass"], dtype=float)
    assert dm.shape == gm.shape, (
        f"trajectory length {dm.shape} != golden {gm.shape}")
    # atol=1e-3 fg sits well above cross-environment float noise (~1e-5 fg
    # observed on CI) and well below any real behavioural drift (a refactor bug
    # diverges by >> 1e-3 fg and grows each tick). rtol=0 keeps it absolute.
    max_dev = float(np.max(np.abs(dm - gm)))
    assert np.allclose(dm, gm, rtol=0.0, atol=1e-3), (
        f"dry_mass trajectory drifted from golden beyond float-noise tolerance "
        f"(max |Δ|={max_dev:.2e} fg, atol=1e-3) — elongation refactor changed behaviour")
