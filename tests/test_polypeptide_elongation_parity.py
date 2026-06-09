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
    elong_rate = []          # smooth, intensive: behaves like dry_mass cross-env
    for _ in range(STEPS):
        c.run(1)
        mass = a["listeners"]["mass"]
        rec.append(round(float(fg_magnitude(mass["dry_mass"])), 6))
        rd = a["listeners"]["ribosome_data"]
        elong_rate.append(round(float(np.asarray(rd["effective_elongation_rate"])), 6))
    bulk = a.get("bulk")
    bulk_sum = int(np.nansum(bulk["count"])) if getattr(bulk, "dtype", None) and bulk.dtype.names else int(np.nansum(bulk))
    return {
        "dry_mass": rec,
        "effective_elongation_rate": elong_rate,
        "bulk_total_at_end": bulk_sum,
    }


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

    # The elongation process's own primary output, not just the mass it feeds.
    # effective_elongation_rate is intensive (~15-18 aa/s) and smooth, so it is
    # as cross-environment-stable as dry_mass — a refactor that perturbs the
    # charging/elongation math but leaves 20-tick mass ~unchanged still shows
    # here. Golden is forward-compatible: skip the check if absent (old golden).
    if "effective_elongation_rate" in golden:
        er = np.asarray(traj["effective_elongation_rate"], dtype=float)
        eg = np.asarray(golden["effective_elongation_rate"], dtype=float)
        er_dev = float(np.max(np.abs(er - eg)))
        assert np.allclose(er, eg, rtol=0.0, atol=1e-3), (
            f"effective_elongation_rate drifted from golden "
            f"(max |Δ|={er_dev:.2e} aa/s, atol=1e-3) — elongation behaviour changed")


# ---------------------------------------------------------------------------
# Model-identity guard. The refactor moved variant selection from a config
# flag (trna_charging=True) to wiring (composites/_helpers.py maps the process
# name -> SteadyStatePolypeptideElongation). A regression that re-routes the
# baseline to the non-charging Base model — e.g. a dropped config key silently
# defaulting to False, or a bad wiring edit — leaves dry_mass *roughly* right
# for many ticks but disables tRNA charging entirely. This asserts the charging
# model is actually live, qualitatively (cross-environment robust: no golden,
# no tight tolerance), so it catches that whole class of "wrong model wired"
# regressions cheaply. Reuses one composite build (~12 s on the behavior job).
@pytest.mark.sim
@pytest.mark.timeout(360, method="signal")
def test_baseline_actually_runs_the_charging_model():
    from v2ecoli import build_composite
    c = build_composite("baseline", cache_dir=CACHE, seed=0)
    a = c.state["agents"]["0"]
    for _ in range(5):
        c.run(1)
    gl = a["listeners"]["growth_limits"]
    pe = a["process_state"]["polypeptide_elongation"]

    # .get() defaults so a missing key (the Base model never writes these)
    # yields the informative assertion below, not a bare KeyError.
    charged_conc = np.asarray(gl.get("charged_trna_conc", []), dtype=float)
    assert charged_conc.size > 0, (
        "charged_trna_conc is absent/empty — baseline is NOT running the "
        "SteadyState charging model (likely fell back to Base/TranslationSupply). "
        "This is the model-selection regression the wiring refactor must prevent.")

    frac = np.asarray(gl.get("fraction_trna_charged", []), dtype=float)
    assert frac.size > 0, "fraction_trna_charged absent — charging model not engaged"
    mean_frac = float(np.nanmean(frac))
    assert 0.3 < mean_frac <= 1.0, (
        f"mean fraction_trna_charged={mean_frac:.3f} is outside the charging "
        f"band (0.3, 1.0]; baseline is not charging tRNAs as SteadyState should")

    gtp = float(np.asarray(pe["gtp_to_hydrolyze"]))
    assert gtp > 0.0, (
        f"gtp_to_hydrolyze={gtp} — no GTP hydrolysis budget; the charging/"
        f"elongation path is not engaged")
