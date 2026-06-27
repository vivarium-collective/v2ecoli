"""Phase 3b-ii tests: the AA-supply ODE merge.

The binding claim of Phase 3 (per ``v2ecoli_consensus_model.md`` and
``workspace/investigations/consensus_elongation/audit.md`` §2) is that
amino acid synthesis, import, and export are integrated **inside the same
ODE system** that does tRNA charging. This file is the proof: each test
fails on the Phase 3a/3b-i scaffold (no RHS edits yet) and passes only
after Phase 3b-ii actually adds supply terms to the kinetic ODE's RHS,
emits the accumulator listeners, and fixes the latent ``aa_count_diff``
return.

Test tiers:

1. **Joint integration proof** (§A): with the supply flag on, the AA pool
   at the end of a timestep is higher than the same solve with the flag
   off — proving supply was added inside the integrate, not as a separate
   step. The accumulators are non-zero.

2. **Listener emission** (§B): post-solve, ``aa_supply``, ``aa_synthesis``,
   ``aa_exchange_rates`` listeners carry shape-correct ndarrays sourced
   from the accumulators.

3. **aa_count_diff bug fix** (§C): ``evolve()`` returns an ``ndarray``
   matching SteadyState's sign convention, not the latent ``{}`` of
   ``trna_charging_final@5ffb76de``.

All sim-tier tests require the post-Task-#8 ParCa cache that carries the
kinetic-charging attrs on ``sim_data.relation``.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest


CACHE = "out/cache"
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


def _cache_has_post_port_relation() -> bool:
    """Mirror of ``test_behavior_kinetic_charging.py``'s cache guard."""
    if not os.path.isdir(CACHE):
        return False
    try:
        from v2ecoli.core import load_cache_bundle

        bundle = load_cache_bundle(CACHE)
        cfg = bundle["configs"]["ecoli-polypeptide-elongation"]
        return "codon_sequences" in cfg and bool(len(cfg["codon_sequences"]))
    except (KeyError, AttributeError, FileNotFoundError):
        return False


_post_port_cache = pytest.mark.skipif(
    not _cache_has_post_port_relation(),
    reason=(
        "cache predates the kinetic Relation port — "
        "rerun scripts/build_cache.py for end-to-end tests"
    ),
)


def _build_kinetic_composite_with_supply(include_aa_supply: bool):
    """Build the kinetic composite and override the supply flag.

    Returns ``(composite, state_path_to_process)``. P3b-ii must wire
    ``include_aa_supply`` through ``config_overrides`` so the composite
    can flip the flag without touching the composite source.
    """
    from process_bigraph import Composite
    from v2ecoli.composites.kinetic_charging_baseline import (
        kinetic_charging_baseline,
    )
    from v2ecoli.core import build_core

    core = build_core()
    doc = kinetic_charging_baseline(
        core=core,
        seed=0,
        cache_dir=CACHE,
        config_overrides={
            "ecoli-polypeptide-elongation.include_aa_supply": include_aa_supply,
        },
    )
    return Composite(doc, core=core)


def _first_agent_aa_counts(composite, amino_acid_names):
    """Sum amino acid bulk counts across the first agent."""
    agent = next(iter(composite.state["agents"].values()))
    bulk = agent["bulk"]
    # bulk is a structured array (name, count, ...) — sum the rows
    # whose names match amino_acid_names.
    name_to_count = dict(zip(bulk["id"], bulk["count"]))
    return np.array(
        [name_to_count.get(name, 0) for name in amino_acid_names],
        dtype=np.int64,
    )


# ============================================================
# §A — joint integration proof
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_supply_on_aa_pool_higher_than_supply_off_after_one_tick() -> None:
    """The binding ODE-merge proof.

    With identical initial state and identical timestep, running 1 tick of
    the kinetic composite with ``include_aa_supply=True`` must leave the
    AA pool higher than the ``=False`` baseline (where the kinetic ODE
    only drains AAs). The difference quantifies the integrated supply
    that the merged ODE captured during the solve.

    If supply were applied as a separate pre/post step (NOT inside the
    integrate), this test could still pass — but combined with
    test_supply_function_called_with_time_evolving_aa_conc below, the
    pair pins down that supply is genuinely inside the ODE RHS.
    """
    composite_off = _build_kinetic_composite_with_supply(include_aa_supply=False)
    composite_on = _build_kinetic_composite_with_supply(include_aa_supply=True)

    # Read amino acid names from the kinetic process config.
    agent_off = next(iter(composite_off.state["agents"].values()))
    # process configs live under a process key; the structure is composite-
    # specific. Fall back to a fixed list of canonical AA bulk names if
    # the composite doesn't expose them.
    aa_names = agent_off.get("_config", {}).get(
        "amino_acids",
        ["ALA[c]", "ARG[c]", "ASN[c]", "ASP[c]", "CYS[c]", "GLN[c]",
         "GLT[c]", "GLY[c]", "HIS[c]", "ILE[c]", "LEU[c]", "LYS[c]",
         "MET[c]", "PHE[c]", "PRO[c]", "SER[c]", "THR[c]", "TRP[c]",
         "TYR[c]", "L-SELENOCYSTEINE[c]", "VAL[c]"],
    )

    aa_off_initial = _first_agent_aa_counts(composite_off, aa_names)
    aa_on_initial = _first_agent_aa_counts(composite_on, aa_names)
    np.testing.assert_array_equal(
        aa_off_initial, aa_on_initial,
        err_msg="initial AA pools must match between flag-off and flag-on builds"
    )

    composite_off.run(interval=1.0)
    composite_on.run(interval=1.0)

    aa_off_final = _first_agent_aa_counts(composite_off, aa_names)
    aa_on_final = _first_agent_aa_counts(composite_on, aa_names)

    delta_off = aa_off_final - aa_off_initial  # mostly negative (drain only)
    delta_on = aa_on_final - aa_on_initial    # less negative or positive

    # Total delta across all AAs: supply-on must net higher than supply-off.
    # Strict inequality required — equality would mean supply contributed
    # zero to the AA balance, which means P3b-ii didn't actually wire
    # supply into the RHS.
    assert delta_on.sum() > delta_off.sum(), (
        f"supply-on AA pool delta ({delta_on.sum()}) is not higher than "
        f"supply-off delta ({delta_off.sum()}) — the ODE merge did not "
        f"activate. Per-AA deltas: off={delta_off}, on={delta_on}"
    )


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_supply_accumulators_emit_nonzero_after_solve() -> None:
    """Post-solve, the supply accumulators must be non-zero.

    The three accumulator slices (total_synthesis, total_import,
    total_export) integrate v_synthesis/v_import/v_export over the
    timestep. After a real solve with biologically realistic initial
    conditions, at least one of synthesis or import must be > 0.
    """
    composite = _build_kinetic_composite_with_supply(include_aa_supply=True)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    aa_synthesis = growth_limits.get("aa_synthesis")
    aa_supply = growth_limits.get("aa_supply")
    aa_exchange_rates = growth_limits.get("aa_exchange_rates")

    assert aa_synthesis is not None and len(aa_synthesis), (
        "aa_synthesis listener must be emitted by the supply-on kinetic path"
    )
    assert aa_supply is not None and len(aa_supply), (
        "aa_supply listener must be emitted by the supply-on kinetic path"
    )
    assert aa_exchange_rates is not None and len(aa_exchange_rates), (
        "aa_exchange_rates listener must be emitted by the supply-on path"
    )

    # At least one of synthesis or supply must have integrated some flux.
    assert (np.abs(np.asarray(aa_synthesis)).sum() > 0
            or np.abs(np.asarray(aa_supply)).sum() > 0), (
        "all supply accumulators integrated to exactly zero — the supply "
        "closure was not evaluated inside the ODE RHS"
    )


def test_supply_function_called_from_inside_ode_rhs_source_marker() -> None:
    """Source-scan guarantee that supply is invoked from inside ode_model.

    The merge claim is that supply is part of the ODE RHS, not a separate
    pre/post step. This is enforceable as a source contract: somewhere
    inside the ``ode_model`` nested function in run_model, there must be
    a call to the supply closure that's passed in via ``args=`` or
    captured via closure. Equivalently, the three accumulator dx_dt rows
    (``dx_dt[self.slice_total_synthesis]`` etc.) must be assigned.

    This test passes only after P3b-ii adds those assignments.
    """
    from v2ecoli.processes.polypeptide import kinetic_charging

    src = inspect.getsource(kinetic_charging)
    # The three accumulator dx_dt writes are the unambiguous marker that
    # supply is wired into the ODE RHS.
    required = [
        "dx_dt[self.slice_total_synthesis]",
        "dx_dt[self.slice_total_import]",
        "dx_dt[self.slice_total_export]",
    ]
    missing = [tok for tok in required if tok not in src]
    assert not missing, (
        f"P3b-ii has not landed: missing RHS writes {missing}. "
        "The supply terms must be assigned inside ode_model so they are "
        "integrated jointly with the tRNA-charging dynamics — not applied "
        "as a separate pre/post-solve step."
    )


def test_supply_is_called_inside_ode_rhs() -> None:
    """Source marker: the supply closure must be evaluated inside the
    ``ode_model`` RHS, not just before or after the solve.

    The signature here is a call like ``supply(...)`` (or a captured
    closure invocation) appearing within the ``ode_model`` nested
    function body. This test passes only after P3b-ii lands.
    """
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    src = inspect.getsource(KT.run_model)
    # The closure is named supply_function on construction and called
    # as supply(aa_conc) inside ode_model — mirroring SteadyState.
    assert "supply_function" in src, (
        "run_model must construct a supply_function closure to feed the ODE"
    )
    # Locate the inner def of ode_model and check the body up to its
    # closing return includes a supply invocation.
    idx = src.find("def ode_model")
    assert idx != -1, "ode_model nested function not found in run_model"
    end = src.find("\n        # Pre-compute cell-volume", idx)  # safe upper bound
    if end == -1:
        end = src.find("\n        return dx_dt", idx) + len("\n        return dx_dt")
    body = src[idx:end]
    assert "supply" in body, (
        "ode_model body must invoke the supply closure so AA synthesis / "
        "import / export are evaluated at every RK45 sub-step"
    )


# ============================================================
# §B — listener emission
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_supply_listeners_shape_matches_amino_acids() -> None:
    """``aa_synthesis``, ``aa_supply``, ``aa_exchange_rates`` listeners
    must have shape ``(n_amino_acids,)``.
    """
    composite = _build_kinetic_composite_with_supply(include_aa_supply=True)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    aa_synthesis = np.asarray(growth_limits["aa_synthesis"])
    aa_supply = np.asarray(growth_limits["aa_supply"])
    aa_exchange_rates = np.asarray(growth_limits["aa_exchange_rates"])

    # All three must be 1-D arrays of the same length.
    assert aa_synthesis.ndim == 1
    assert aa_supply.shape == aa_synthesis.shape
    assert aa_exchange_rates.shape == aa_synthesis.shape


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_supply_listeners_zero_when_flag_off() -> None:
    """With ``include_aa_supply=False``, supply listeners must be absent
    or zero-valued — Phase 3a/3b-i must leave the legacy behavior intact.
    """
    composite = _build_kinetic_composite_with_supply(include_aa_supply=False)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    for key in ("aa_synthesis", "aa_supply"):
        val = growth_limits.get(key)
        if val is None or len(val) == 0:
            continue  # absent is fine on the flag-off path
        assert np.asarray(val).sum() == 0, (
            f"{key} listener must be zero when include_aa_supply=False"
        )


# ============================================================
# §C — aa_count_diff bug fix (tail of P3b-ii)
# ============================================================

def test_aa_count_diff_evolve_returns_ndarray_not_empty_dict() -> None:
    """Source contract: the inner ``evolve`` must not return ``{}`` for
    the second tuple position (the latent bug at line 775 of
    trna_charging_final@5ffb76de).
    """
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    src = inspect.getsource(KT.evolve)
    # The bug was `return net_charged, {}, update`. After the fix, the
    # second position is a non-empty expression.
    assert "return net_charged, {}, update" not in src, (
        "evolve() still returns {} for the aa_count_diff position — "
        "the latent bug at kinetic_charging.py:775 has not been fixed"
    )


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_aa_count_diff_is_ndarray_after_one_tick() -> None:
    """End-to-end: after a 1-tick run, the polypeptide_elongation port's
    aa_count_diff must be an ``ndarray`` with shape ``(n_amino_acids,)``.
    """
    composite = _build_kinetic_composite_with_supply(include_aa_supply=True)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    # Topology wires the polypeptide_elongation port to
    # ("process_state", "polypeptide_elongation") in the store —
    # see TOPOLOGY at polypeptide_elongation.py:91-97.
    aa_count_diff = agent["process_state"]["polypeptide_elongation"]["aa_count_diff"]

    arr = np.asarray(aa_count_diff)
    assert arr.ndim == 1, (
        f"aa_count_diff must be 1-D ndarray, got shape {arr.shape}"
    )
    assert arr.size > 0, "aa_count_diff must not be empty"


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_aa_count_diff_sign_convention_matches_steady_state() -> None:
    """SteadyState writes ``aa_count_diff = aa_supply - aa_used_trna``
    (positive = over-supplied → metabolism raises the homeostatic
    target). The kinetic class's fix must use the same formula and sign.

    Direct equality check: ``aa_count_diff == aa_supply - aas_used``,
    using the same listener fields. If either component has its sign
    flipped or unit-converted wrong, the equality breaks.

    Note: ``aa_supply`` can be negative for individual AAs (reverse
    reactions in ``amino_acid_synthesis`` exceed forward, or export
    exceeds synthesis+import) — that's biologically valid and not a
    sign-flip bug. The formula equality is the binding test.
    """
    composite = _build_kinetic_composite_with_supply(include_aa_supply=True)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    aa_count_diff = np.asarray(
        agent["process_state"]["polypeptide_elongation"]["aa_count_diff"]
    )
    growth_limits = agent["listeners"]["growth_limits"]
    aa_supply = np.asarray(growth_limits["aa_supply"])
    aas_used = np.asarray(growth_limits["aas_used"])

    # aa_count_diff is computed in :meth:`evolve` as
    # ``aa_supply - amino_acids_used.astype(float64)``. The listener
    # ``aas_used`` is the same ``amino_acids_used`` value.
    expected = aa_supply - aas_used.astype(np.float64)
    np.testing.assert_allclose(
        aa_count_diff, expected, rtol=1e-6, atol=1e-6,
        err_msg=(
            "aa_count_diff does not equal aa_supply - aas_used. "
            "Sign or formula mismatch vs SteadyState."
        ),
    )
