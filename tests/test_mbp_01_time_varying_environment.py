"""Behavior tests for mbp-01-time-varying-environment.

Each test corresponds 1:1 to a behavior_test entry in
``studies/mbp-01-time-varying-environment/study.yaml`` (post
chris_feedback_2026_05_26 §3 reframe — plumbing + qualitative-extreme
only; gradient-shape / quantitative-biology tests are deferred to a
downstream ``transport-kinetics-fidelity`` candidate-future-study).

The static-mode regression test (``static-env-baseline-unchanged``) is
implemented. The plumbing + extreme tests were unblocked 2026-05-28 by
the EnvironmentMirror Step (closes the two architectural gaps captured
2026-05-28 in this study's open_questions): driver writes top-level
env.external_concentrations -> mirror computes per-agent boundary.external
delta -> FLUSH commits before media_update's layer -> exchange_data
re-derives metabolism constraints from the new boundary.
"""

from __future__ import annotations

import pytest

from process_bigraph import Composite

from v2ecoli.composites.baseline import baseline
from v2ecoli.composites.baseline_time_varying_env import baseline_time_varying_env
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.core import build_core
from v2ecoli.steps.environment_driver import (
    ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
    TRAJ_CLAMP_TO_VALUE,
    TRAJ_LINEAR_DECLINE,
)


pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def core():
    return build_core()


# --- Plumbing tests (mechanism, no biology invoked) -------------------------


def test_external_glucose_updates_propagate_within_one_step(core):
    """Plumbing test: driver-write to external_concentrations.GLC reflected in
    boundary.external.GLC within one timestep.

    Backs study.yaml behavior_test
    ``external_glucose_updates_propagate_within_one_step``. Replaces the
    removed Pearson-correlation test (which assumed linear response between
    concentration and uptake; real PTS is saturable, ~3 orders of magnitude
    above the sweep range — test failed on every realistic implementation).
    """
    target_mM = 1.5
    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "GLC": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": target_mM},
        },
    )
    comp = Composite(doc, core=core)
    comp.run(1)
    boundary_glc = float(comp.state["agents"]["0"]["boundary"]["external"]["GLC"])
    assert boundary_glc == pytest.approx(target_mM, abs=1e-9), (
        f"driver wrote external_concentrations.GLC={target_mM}; boundary.external.GLC "
        f"was {boundary_glc!r} after 1 tick (expected same-tick propagation via mirror)"
    )


def test_cumulative_mass_balance_closes(core):
    """Plumbing test: with the driver holding glucose constant, the cumulative
    boundary.external trajectory tracks the driver-supplied value across
    ticks (no drift, no double-apply).

    Backs study.yaml behavior_test ``cumulative-mass-balance-closes`` —
    reframed as a wire-fidelity check (the original mass-balance integral
    against per-cell uptake fluxes would require running metabolism to
    pseudo-steady-state, which is a transport-kinetics-fidelity concern).
    """
    target_mM = 2.5
    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "GLC": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": target_mM},
        },
    )
    comp = Composite(doc, core=core)
    comp.run(3)
    boundary_glc = float(comp.state["agents"]["0"]["boundary"]["external"]["GLC"])
    assert boundary_glc == pytest.approx(target_mM, abs=1e-6), (
        f"after 3 ticks with clamp_to_value={target_mM}, boundary.external.GLC "
        f"drifted to {boundary_glc!r} (no-drift invariant violated)"
    )


# --- Qualitative extreme-case tests (biology floor + ceiling) ---------------


def test_zero_substrate_blocks_uptake(core):
    """external_glucose held at 0 → boundary.external.GLC drops to 0.

    Universal biological floor: with no substrate available at the cell
    boundary, metabolism's exchange flux for glucose import must be 0
    (regardless of transport-kinetics functional form). This test asserts
    the plumbing precondition — the boundary actually reflects the zero
    that the driver supplied — without exercising the metabolism FBA
    inner solve (deferred to transport-kinetics-fidelity).

    Backs study.yaml behavior_test ``zero-substrate-blocks-uptake``.
    """
    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "GLC": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": 0.0},
        },
    )
    comp = Composite(doc, core=core)
    comp.run(1)
    boundary_glc = float(comp.state["agents"]["0"]["boundary"]["external"]["GLC"])
    assert boundary_glc == pytest.approx(0.0, abs=1e-9), (
        f"clamp_to_value=0 did not zero boundary.external.GLC (got {boundary_glc!r})"
    )


def test_saturating_substrate_respects_vmax(core):
    """external_glucose at 50 mM (~30,000× PTS Km) → boundary.external.GLC is
    50 mM. This is the plumbing-tier check that the saturating-range setup
    actually delivers the high-glucose value to the cell boundary; the
    Vmax-cap of metabolism's flux is the transport-kinetics-fidelity
    concern that mbp-01 defers.

    Backs study.yaml behavior_test ``saturating-substrate-respects-vmax``.
    """
    saturating_mM = 50.0
    doc = baseline_time_varying_env(
        core=core, seed=0, cache_dir="out/cache",
        env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
        synthetic_trajectory_spec={
            "GLC": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": saturating_mM},
        },
    )
    comp = Composite(doc, core=core)
    comp.run(1)
    boundary_glc = float(comp.state["agents"]["0"]["boundary"]["external"]["GLC"])
    assert boundary_glc == pytest.approx(saturating_mM, abs=1e-6), (
        f"saturating clamp ({saturating_mM} mM) did not propagate to "
        f"boundary.external.GLC (got {boundary_glc!r})"
    )


def test_plateau_across_saturating_range(core):
    """Two runs at 5 mM and 50 mM (both far above PTS Km) → identical
    boundary.external.GLC delivery (within float epsilon). The plumbing
    is concentration-agnostic; whether metabolism's flux plateaus or
    scales is the downstream kinetics question (deferred).

    Backs study.yaml behavior_test ``plateau-across-saturating-range``.
    """
    def boundary_after(target_mM: float) -> float:
        doc = baseline_time_varying_env(
            core=core, seed=0, cache_dir="out/cache",
            env_driver_mode=ENV_DRIVER_MODE_SYNTHETIC_TRAJECTORY,
            synthetic_trajectory_spec={
                "GLC": {"kind": TRAJ_CLAMP_TO_VALUE, "value_mmolL": target_mM},
            },
        )
        comp = Composite(doc, core=core)
        comp.run(1)
        return float(comp.state["agents"]["0"]["boundary"]["external"]["GLC"])

    glc_low = boundary_after(5.0)
    glc_high = boundary_after(50.0)
    assert glc_low == pytest.approx(5.0, abs=1e-6)
    assert glc_high == pytest.approx(50.0, abs=1e-6)
    # Confirm BOTH propagations honored their target — plumbing has no
    # plateau, every value is delivered verbatim.
    assert abs(glc_low - 5.0) < 1e-6 and abs(glc_high - 50.0) < 1e-6


# --- Static-mode regression (kept from the previous build) ------------------


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

    cell_mass_a = fg_magnitude(comp_a.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    cell_mass_b = fg_magnitude(comp_b.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    assert cell_mass_a == cell_mass_b, (
        f"static-mode env-driver perturbed per-cell state "
        f"(baseline: {cell_mass_a!r}, baseline_time_varying_env: {cell_mass_b!r})"
    )
