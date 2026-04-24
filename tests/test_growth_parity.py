"""Growth-rate parity regression: pin baseline mass trajectory vs vEcoli.

A whole family of bugs in this codebase quietly shifts biomass production
without throwing an error:

  - the unum→pint migration + stale ParCa cache silently dropped
    ``ecoli-polypeptide-elongation`` from save_cache's output, which
    starved the FBA scheduler and caused a runaway refire loop;
  - the ``get_biomass_as_concentrations`` shim in LoadSimData dropped the
    ``rp_ratio`` kwarg, leaving 30 metabolite targets inflated by ~15%
    and small-molecule submass growing 13% faster than vEcoli over a
    cell cycle.

Neither fires as a crash. Both show up as drift against vEcoli's mass
trajectory. This test pins that trajectory: at three checkpoints during
the first generation, each submass is within a tight tolerance of the
value a fresh vEcoli 1.0 run produces for the same seed and media.

The golden values below were captured on 2026-04-19 from
``python scripts/run_vecoli_v1.py`` against the v1_v2_report reference
ParCa fixture (``models/parca/parca_state.pkl.gz``). If you intentionally
change biology (new processes, new KB), recapture them — but document
the diff in the PR, since this is the regression gate for silent
metabolism / translation drift.
"""
from __future__ import annotations

import os

import pytest

# Side-effect import: registers `nucleotide` / `amino_acid` / `count` on the
# shared pint registry before dill.load of the cache.
import v2ecoli.library.unit_bridge  # noqa: F401


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir('out/cache') and not os.environ.get('CI'),
        reason="cache dir 'out/cache' not present; "
               "rebuild with `python scripts/build_cache.py`",
    ),
]

# (time_s, {submass_name: (golden_value_fg, relative_tolerance)})
#
# Tolerances are calibrated so the known regressions fail and the current
# ~0-3% v2ecoli-vs-vEcoli drift passes:
#   - rp_ratio bug: dry +7%, small_mol +13% — both beyond 5% / 7%
#   - refire loop: doesn't reach these sim times before OOM
#   - current v2ecoli drift: ≤1.3% dry, ≤3.3% small_mol
GOLDEN = [
    (500, {
        'dry_mass':     (425.5, 0.05),
        'protein_mass': (206.9, 0.05),
        'rna_mass':     ( 56.6, 0.05),
        'dna_mass':     (  7.9, 0.05),
        'small_mol':    (153.5, 0.07),
    }),
    (1000, {
        'dry_mass':     (483.4, 0.05),
        'protein_mass': (230.4, 0.05),
        'rna_mass':     ( 64.5, 0.05),
        'dna_mass':     (  8.9, 0.05),
        'small_mol':    (179.0, 0.07),
    }),
    (1500, {
        'dry_mass':     (547.7, 0.05),
        'protein_mass': (257.5, 0.05),
        'rna_mass':     ( 73.3, 0.05),
        'dna_mass':     (  9.5, 0.05),
        'small_mol':    (206.8, 0.07),
    }),
]


def _extract_submasses(mass: dict) -> dict[str, float]:
    """Collapse the mass listener into the five submasses the golden
    references pin."""
    return {
        'dry_mass':     float(mass.get('dry_mass', 0.0) or 0.0),
        'protein_mass': float(mass.get('protein_mass', 0.0) or 0.0),
        'rna_mass': (
            float(mass.get('rRna_mass', 0.0) or 0.0)
            + float(mass.get('tRna_mass', 0.0) or 0.0)
            + float(mass.get('mRna_mass', 0.0) or 0.0)
        ),
        'dna_mass':     float(mass.get('dna_mass', 0.0) or 0.0),
        'small_mol':    float(mass.get('smallMolecule_mass', 0.0) or 0.0),
    }


@pytest.mark.timeout(600)
def test_baseline_growth_trajectory_matches_vecoli(sim_data_cache):
    """Baseline cell's submass trajectory stays within tolerance of vEcoli.

    Runs one baseline composite from t=0 to t=1500 (pre-division), sampling
    submasses at 500s, 1000s, 1500s. Each submass must be within the per-
    component tolerance of the golden reference captured from
    ``scripts/run_vecoli_v1.py`` on 2026-04-19.

    Failure modes this catches:
      - Metabolism producing too much / too little biomass (the rp_ratio
        bug — 15% target inflation → 13% small-mol drift).
      - Translation / transcription rates silently shifting (would show
        up in protein_mass or rna_mass first).
      - Any silent-drop of a critical process config (would show up as
        near-zero growth or protein stuck at initial).

    Running to 1500s rather than the full 2520s cell cycle is a deliberate
    cost/coverage trade — 1500s is pre-division so no daughter handling,
    and the three checkpoints span the first three-quarters of the cycle
    which is where drift becomes visible.
    """
    from v2ecoli.composite import make_composite

    composite = make_composite(cache_dir='out/cache', seed=0)

    sim_time_so_far = 0
    for target_t, expected in GOLDEN:
        composite.run(target_t - sim_time_so_far)
        sim_time_so_far = target_t

        cell = composite.state.get('agents', {}).get('0')
        assert cell is not None, (
            f'agent 0 missing at t={target_t}s — early division or '
            f'composite teardown? Divided mid-run should not happen '
            f'before t≈2200s.'
        )

        actual = _extract_submasses(cell.get('listeners', {}).get('mass', {}))

        failures = []
        for key, (golden, tol) in expected.items():
            v = actual[key]
            rel_err = abs(v - golden) / golden
            if rel_err > tol:
                failures.append(
                    f'  {key} at t={target_t}s: {v:.1f} vs vEcoli {golden:.1f} '
                    f'(diff {rel_err*100:.1f}%, tolerance {tol*100:.0f}%)'
                )
        assert not failures, (
            f'v2ecoli mass trajectory drifted from vEcoli at t={target_t}s:\n'
            + '\n'.join(failures)
            + '\n(see tests/test_growth_parity.py docstring for how to recapture goldens.)'
        )
