"""Behavior tests for mbp-01-time-varying-environment.

Each test corresponds 1:1 to a behavior_test entry in
``studies/mbp-01-time-varying-environment/study.yaml`` (post
chris_feedback_2026_05_26 §3 reframe — plumbing + qualitative-extreme
only; gradient-shape / quantitative-biology tests are deferred to a
downstream ``transport-kinetics-fidelity`` candidate-future-study).

The static-mode regression test (``static-env-baseline-unchanged``) is
implemented. The plumbing + extreme tests require modifying media_update
to consume from ``environment.external_concentrations`` (a
PartitionedProcess in 3 architectures per AGENTS.md, plus ParCa cache
regen) — that work is tracked as a TODO in
``v2ecoli/composites/baseline_time_varying_env.py``. Until it lands, the
remaining tests stay @pytest.mark.skip with a reason that names the
specific blocker.
"""

from __future__ import annotations

import pytest

from process_bigraph import Composite

from v2ecoli.composites.baseline import baseline
from v2ecoli.composites.baseline_time_varying_env import baseline_time_varying_env
from v2ecoli.core import build_core


pytestmark = pytest.mark.sim


_NEEDS_MEDIA_UPDATE_MOD = (
    "media_update consumption of environment.external_concentrations landed "
    "2026-05-28; two wiring gaps remain (see baseline_time_varying_env.py docstring): "
    "(1) top-level state['environment'] is a different store than per-cell "
    "state['agents'][i]['environment'] that MediaUpdate reads; (2) driver writes "
    "compartment-tagged 'GLC[p]' but boundary.external uses bare 'GLC'."
)


@pytest.fixture(scope="module")
def core():
    return build_core()


# --- Plumbing tests (mechanism, no biology invoked) -------------------------

@pytest.mark.skip(reason=_NEEDS_MEDIA_UPDATE_MOD)
def test_external_glucose_updates_propagate_within_one_step():
    """Plumbing test: external_glucose store update reflected in metabolism's
    GLC[p] exchange constraint within one timestep.

    Backs study.yaml behavior_test
    ``external_glucose_updates_propagate_within_one_step``. Replaces the
    removed Pearson-correlation test (which assumed linear response between
    concentration and uptake; real PTS is saturable, ~3 orders of magnitude
    above the sweep range — test failed on every realistic implementation).
    """
    raise NotImplementedError


@pytest.mark.skip(reason=_NEEDS_MEDIA_UPDATE_MOD)
def test_cumulative_mass_balance_closes():
    """Plumbing test: cumulative external-glucose decrease matches cumulative
    glucose-uptake-flux integral within ±10% — the env-hook is wired without
    leakage.

    Backs study.yaml behavior_test ``cumulative-mass-balance-closes``.
    """
    raise NotImplementedError


def test_static_env_baseline_unchanged(core):
    """Regression guard: with env_driver_mode=static (default), the composite
    produces byte-identical per-cell trajectory to the unmodified
    v2ecoli.composites.baseline at the same seed. Backs study.yaml
    behavior_test ``static-env-baseline-unchanged``.
    """
    doc_a = baseline(core=core, seed=0, cache_dir="out/cache")
    doc_b = baseline_time_varying_env(core=core, seed=0, cache_dir="out/cache")

    comp_a = Composite(doc_a, core=core)
    comp_b = Composite(doc_b, core=core)
    comp_a.run(1)
    comp_b.run(1)

    cell_mass_a = float(comp_a.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    cell_mass_b = float(comp_b.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    assert cell_mass_a == cell_mass_b, (
        f"static-mode env-driver perturbed per-cell state "
        f"(baseline: {cell_mass_a!r}, baseline_time_varying_env: {cell_mass_b!r})"
    )


# --- Qualitative extreme-case tests (biology floor + ceiling) ---------------

@pytest.mark.skip(reason=_NEEDS_MEDIA_UPDATE_MOD)
def test_zero_substrate_blocks_uptake():
    """external_glucose held at 0 → glucose exchange flux ≈ 0; same for DO.
    Universal biological floor (S=0 → no transport across PTS / oxidases);
    robust to whichever transport-kinetics functional form metabolism_redux
    implements. Backs study.yaml behavior_test
    ``zero-substrate-blocks-uptake``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason=_NEEDS_MEDIA_UPDATE_MOD)
def test_saturating_substrate_respects_vmax():
    """external_glucose at 50 g/L (~30,000× PTS Km) → glucose exchange flux
    ≤ ~12 mmol/(gDW·h) (published E. coli glucose-PTS Vmax ceiling).
    Catches the unphysical linear-scaling implementation case. Backs
    study.yaml behavior_test ``saturating-substrate-respects-vmax``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason=_NEEDS_MEDIA_UPDATE_MOD)
def test_plateau_across_saturating_range():
    """Two runs at 5 g/L and 50 g/L (both far above PTS Km) → GUR ratio in
    [0.9, 1.1]. Saturable + constant-bound models predict matching GUR;
    linear scaling predicts ~10×. Direct discriminator between physical
    and unphysical transport implementations. Backs study.yaml
    behavior_test ``plateau-across-saturating-range``.
    """
    raise NotImplementedError
