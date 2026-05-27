"""Behavior tests for mbp-01-time-varying-environment.

Scaffold for Build phase. Each test corresponds 1:1 to a behavior_test
entry in ``studies/mbp-01-time-varying-environment/study.yaml`` (post
chris_feedback_2026_05_26 §3 reframe — plumbing + qualitative-extreme
only; gradient-shape / quantitative-biology tests are deferred to a
downstream ``transport-kinetics-fidelity`` candidate-future-study).

The tests are CURRENTLY SKIPPED with the reason "Build-phase scaffold —
EnvironmentDriver wire-up TODO" so CI doesn't go red on the in-progress
work. As each TODO in
``v2ecoli/composites/baseline_time_varying_env.py`` lands, the
corresponding ``@pytest.mark.skip`` decorator should be removed.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.sim


# --- Plumbing tests (mechanism, no biology invoked) -------------------------

@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
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


@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
def test_cumulative_mass_balance_closes():
    """Plumbing test: cumulative external-glucose decrease matches cumulative
    glucose-uptake-flux integral within ±10% — the env-hook is wired without
    leakage.

    Backs study.yaml behavior_test ``cumulative-mass-balance-closes``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
def test_static_env_baseline_unchanged():
    """Regression guard: with env_driver_mode=static, the composite produces
    byte-identical trajectory (DnaA count, growth rate) to the unmodified
    v2ecoli.composites.baseline. Backs study.yaml behavior_test
    ``static-env-baseline-unchanged``.
    """
    raise NotImplementedError


# --- Qualitative extreme-case tests (biology floor + ceiling) ---------------

@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
def test_zero_substrate_blocks_uptake():
    """external_glucose held at 0 → glucose exchange flux ≈ 0; same for DO.
    Universal biological floor (S=0 → no transport across PTS / oxidases);
    robust to whichever transport-kinetics functional form metabolism_redux
    implements. Backs study.yaml behavior_test
    ``zero-substrate-blocks-uptake``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
def test_saturating_substrate_respects_vmax():
    """external_glucose at 50 g/L (~30,000× PTS Km) → glucose exchange flux
    ≤ ~12 mmol/(gDW·h) (published E. coli glucose-PTS Vmax ceiling).
    Catches the unphysical linear-scaling implementation case. Backs
    study.yaml behavior_test ``saturating-substrate-respects-vmax``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — EnvironmentDriver wire-up TODO")
def test_plateau_across_saturating_range():
    """Two runs at 5 g/L and 50 g/L (both far above PTS Km) → GUR ratio in
    [0.9, 1.1]. Saturable + constant-bound models predict matching GUR;
    linear scaling predicts ~10×. Direct discriminator between physical
    and unphysical transport implementations. Backs study.yaml
    behavior_test ``plateau-across-saturating-range``.
    """
    raise NotImplementedError
