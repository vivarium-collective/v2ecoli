"""Behavior test: the baseline cell conserves mass.

Over a run, the change in total cell mass must equal the net mass exchanged
across the environment boundary by metabolism (the only mass source/sink) —
every other process merely repackages atoms it already holds. This is the
runtime form of AGENTS.md conservation check #4: a silent leak in any process
(creating or destroying mass) shows up as cumulative drift between Δcell_mass
and the net metabolic exchange.

Implementation: enable the opt-in ``mass_conservation`` feature (which wires the
``MassConservationDeriver``), run the baseline, and assert the **steady-state**
relative drift stays small.

Why a window, not the whole run: the cumulative residual carries a fixed
~1.2 fg offset established during cell initialization (the first metabolic
solves / mass-listener spin-up), which does NOT accumulate afterward — per-tick
conservation is excellent once the cell is going. Measuring from start would
divide that one-time offset by a small early Δcell and spuriously flag drift, so
the test measures the residual CHANGE over a post-warmup window normalized by
the cell-mass gain over that window. A healthy baseline drifts ~1% there; the
gate is 3%.

Marked ``sim`` — it runs a live simulation (no cached trajectory, because the
conservation residual is only produced when the feature is wired).
"""

import pytest

from v2ecoli.library.quantity_helpers import fg_magnitude

WARMUP_TICKS = 20      # discard the cell-initialization transient
WINDOW_TICKS = 80      # steady-state window over which conservation is gated
CONSERVATION_TOLERANCE = 0.03  # ~1% healthy; 3% leaves headroom, catches leaks


def _mass_node(state, depth=0):
    if not isinstance(state, dict) or depth > 14:
        return None
    if "cell_mass" in state and "conservation_relative_cumulative" in state:
        return state
    for v in state.values():
        node = _mass_node(v, depth + 1)
        if node is not None:
            return node
    return None


def _read(comp):
    node = _mass_node(comp.state)
    assert node is not None, (
        "mass_conservation feature did not wire — no conservation_relative_"
        "cumulative emitted on listeners.mass.")
    return (fg_magnitude(node["cell_mass"]),
            fg_magnitude(node["conservation_residual_cumulative"]))


@pytest.mark.sim
@pytest.mark.parametrize("seed", [0])
def test_baseline_conserves_mass(sim_data_cache, seed):
    """Over a steady-state window, the cell's mass gain matches the net
    metabolic boundary exchange to within a few percent.

    Single representative seed: mass conservation is a physical invariant of the
    model, not a stochastic property, so one seed exercises it fully. (Kept to
    one seed to hold the CI behavior-tests job near its pre-units runtime — the
    coercing-units overhead already slows every sim; a multi-seed sweep belongs
    in the nightly @slow suite.)"""
    import v2ecoli
    from v2ecoli.composites.baseline import enable_features

    enable_features("mass_conservation")
    try:
        comp = v2ecoli.build_composite("baseline", seed=seed, cache_dir=sim_data_cache)
        comp.run(WARMUP_TICKS + 1)            # +1: first tick is the listener baseline
        m_start, resid_start = _read(comp)
        comp.run(WINDOW_TICKS)
        m_end, resid_end = _read(comp)
    finally:
        enable_features()  # clear the module-global opt-in for other tests

    growth = m_end - m_start                  # cell-mass gain over the window
    window_residual = resid_end - resid_start  # unbalanced mass over the window
    rel = abs(window_residual) / growth if growth else float("inf")

    # Guard against a trivially-passing stalled cell.
    assert growth > 0, (
        f"cell did not grow over the window ({m_start:.1f} -> {m_end:.1f} fg) — "
        f"conservation drift is not meaningful on a stalled run.")

    assert rel < CONSERVATION_TOLERANCE, (
        f"[seed {seed}] baseline steady-state mass-conservation drift {rel:.3f} "
        f"exceeds {CONSERVATION_TOLERANCE:.0%} over {WINDOW_TICKS} ticks: a process "
        f"is creating or destroying mass without a justified source/sink. window "
        f"residual = {window_residual:.3g} fg over {growth:.1f} fg of cell-mass gain.")


# Division conserves mass to within this fraction (it is a count partition with
# no environment exchange, so total mass should be essentially exact).
DIVISION_TOLERANCE = 0.02


@pytest.mark.sim
def test_division_conserves_mass(predivision_state, sim_data_cache):
    """Division is a partition of the mother cell into two daughters with NO
    environment exchange — so total cell mass must be conserved:
    mother_cell_mass ≈ daughter1 + daughter2. Complements the growth-phase
    conservation test (this is the division-event invariant)."""
    from process_bigraph import Composite
    from v2ecoli.library.division import divide_cell
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline, seed_mass_listener

    core = build_core()

    def _cell_mass(state):
        # Build the full document, swap in the given biological state, re-seed
        # the mass listener against it, and read the resulting cell mass.
        doc = baseline(core=core, seed=0, cache_dir=sim_data_cache)
        agent = doc["state"]["agents"]["0"]
        for key in ("bulk", "unique", "environment", "boundary"):
            if key in state:
                agent[key] = state[key]
        agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
        seed_mass_listener(agent, core)
        comp = Composite(doc, core=core)
        return fg_magnitude(comp.state["agents"]["0"]["listeners"]["mass"]["cell_mass"])

    mother = _cell_mass(predivision_state)
    d1_state, d2_state = divide_cell(predivision_state)
    m1, m2 = _cell_mass(d1_state), _cell_mass(d2_state)
    total = m1 + m2
    rel = abs(total - mother) / mother if mother else float("inf")

    assert rel < DIVISION_TOLERANCE, (
        f"division did not conserve mass: mother={mother:.1f} fg vs "
        f"daughters {m1:.1f}+{m2:.1f}={total:.1f} fg (rel {rel:.3f}). Division is "
        f"a count partition with no exchange — total mass must be preserved.")
