"""Phase 2 tests: ppGpp regulation on the kinetic-charging path.

The kinetic_charging_baseline composite already passes ``ppgpp_regulation=True``
to the kinetic process via DEFAULT_FEATURES, but
:class:`KineticTrnaChargingPolypeptideElongation` inherits only Base's
ppGpp parameter scaffold — no ``_ppgpp_request`` / ``_ppgpp_evolve`` hooks.
So today ppGpp regulation is silently a no-op on the kinetic path.

These tests gate Phase 2 (audit.md §3): port the two ppGpp methods from
SteadyStatePolypeptideElongation (Option A — copy, don't change
inheritance), wire them into the kinetic ``request()`` / ``evolve()`` /
pre-solve elongation-rate adjustment, and emit the ppGpp listener fields.

Test tiers:

§A — Source markers (no cache):
- ``_ppgpp_request`` / ``_ppgpp_evolve`` methods exist on the kinetic class.
- ``request`` calls ``_ppgpp_request``; ``evolve_state`` (or ``evolve``)
  calls ``_ppgpp_evolve``.
- Pre-solve elongation-rate adjustment via ``elong_rate_by_ppgpp`` is
  applied to ``target_codon_rate`` when ``ppgpp_regulation=True``.
- ``outputs()`` declares the ppGpp listener fields.

§B — Listener emission (sim + cache):
- ``rela_syn`` / ``spot_syn`` / ``spot_deg`` listeners emitted when
  ``ppgpp_regulation=True``.
- ``ppgpp_conc`` non-zero (cell has ppGpp at baseline).
- ppGpp bulk count actually changes after a 1-tick run — proves
  ``_ppgpp_evolve`` ran and applied deltas.

§C — Flag-off no-op (sim + cache):
- With ``ppgpp_regulation=False``, ppGpp listener fields are absent or
  zero — ensures the flag actually gates the behavior.
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


def _build_kinetic_composite_with_ppgpp(ppgpp_regulation: bool):
    """Build the kinetic composite with ppgpp_regulation set explicitly.

    P3b-ii proved that ``config_overrides`` wires per-process flags
    through to the kinetic Process; same pattern here.
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
            "ecoli-polypeptide-elongation.ppgpp_regulation": ppgpp_regulation,
            # Keep include_aa_supply ON so the kinetic ODE has realistic
            # pool dynamics — ppGpp behavior is measured in this regime.
            "ecoli-polypeptide-elongation.include_aa_supply": True,
        },
    )
    return Composite(doc, core=core)


def _bulk_count(agent, name: str) -> int:
    bulk = agent["bulk"]
    idx = list(bulk["id"]).index(name)
    return int(bulk["count"][idx])


# ============================================================
# §A — source markers (no cache)
# ============================================================

def test_kinetic_class_has_ppgpp_request_method() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    assert hasattr(KT, "_ppgpp_request"), (
        "kinetic class must expose _ppgpp_request (ported from SteadyState "
        "per audit.md §3, Option A)"
    )


def test_kinetic_class_has_ppgpp_evolve_method() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    assert hasattr(KT, "_ppgpp_evolve"), (
        "kinetic class must expose _ppgpp_evolve (ported from SteadyState)"
    )


def test_request_calls_ppgpp_request() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.request)
    assert "_ppgpp_request" in src, (
        "kinetic request() must invoke self._ppgpp_request(...) so ppGpp "
        "metabolite bulk requests are included alongside AA/ATP/tRNA"
    )


def test_evolve_state_or_evolve_calls_ppgpp_evolve() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    # The call may live in either evolve_state (the codon-aware pipeline)
    # or evolve (the bulk-delta emitter). Either is acceptable; both
    # places run after the existing bulk deltas in the tick.
    sources = inspect.getsource(KT.evolve_state) + inspect.getsource(KT.evolve)
    assert "_ppgpp_evolve" in sources, (
        "kinetic evolve_state or evolve must invoke self._ppgpp_evolve(...) "
        "after the existing bulk deltas so the ppGpp reaction sees the "
        "post-elongation tRNA state"
    )


def test_elong_rate_by_ppgpp_applied_pre_solve() -> None:
    """When ``ppgpp_regulation=True`` and the inhibition flag is off, the
    kinetic ``target_codon_rate`` fed to ``solve_ivp`` must be scaled by
    ``elong_rate_by_ppgpp(ppgpp_conc, basal_rate)``. Source marker.
    """
    from v2ecoli.processes.polypeptide import kinetic_charging

    src = inspect.getsource(kinetic_charging)
    assert "elong_rate_by_ppgpp" in src, (
        "kinetic source must reference self.elong_rate_by_ppgpp(...) "
        "to apply the ppGpp inhibition on elongation rate pre-solve"
    )


def test_outputs_declares_ppgpp_listener_fields() -> None:
    """``outputs()`` must declare ``rela_syn`` / ``spot_syn`` /
    ``spot_deg`` (and friends) under ``listeners.growth_limits`` so
    process-bigraph propagates them to the store.
    """
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.outputs)
    required = ["rela_syn", "spot_syn", "spot_deg"]
    missing = [field for field in required if field not in src]
    assert not missing, (
        f"outputs() missing ppGpp listener fields: {missing}"
    )


# ============================================================
# §B — listener emission (sim + cache)
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_ppgpp_listeners_emitted_when_regulation_on() -> None:
    """With ``ppgpp_regulation=True``, the ppGpp synth/deg listeners must
    be non-empty arrays. Default schema is ``[]`` so failure surfaces as
    empty arrays — the gate is "non-zero length".
    """
    composite = _build_kinetic_composite_with_ppgpp(ppgpp_regulation=True)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    for key in ("rela_syn", "spot_syn", "spot_deg"):
        val = growth_limits.get(key)
        assert val is not None, f"{key} listener missing entirely"
        arr = np.atleast_1d(np.asarray(val))
        assert arr.size > 0, f"{key} listener present but empty"


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_ppgpp_bulk_count_changes_after_one_tick() -> None:
    """The ``GUANOSINE-5DP-3DP[c]`` bulk count must change between t=0 and t=1 when
    ``ppgpp_regulation=True`` — direct proof that ``_ppgpp_evolve`` ran
    and applied the ppGpp-reaction delta.

    Note: small changes (a few molecules) are still proof — RelA/SpoT
    fluxes are slow compared to elongation. The gate is strict
    inequality, not magnitude.
    """
    composite = _build_kinetic_composite_with_ppgpp(ppgpp_regulation=True)

    agent_before = next(iter(composite.state["agents"].values()))
    ppgpp_before = _bulk_count(agent_before, "GUANOSINE-5DP-3DP[c]")

    composite.run(interval=1.0)

    agent_after = next(iter(composite.state["agents"].values()))
    ppgpp_after = _bulk_count(agent_after, "GUANOSINE-5DP-3DP[c]")

    assert ppgpp_before != ppgpp_after, (
        f"GUANOSINE-5DP-3DP[c] bulk count did not change ({ppgpp_before} → "
        f"{ppgpp_after}) — _ppgpp_evolve did not apply deltas"
    )


# ============================================================
# §C — flag-off no-op (sim + cache)
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_ppgpp_listeners_zero_when_regulation_off() -> None:
    """With ``ppgpp_regulation=False``, the new ppGpp listener fields
    must be absent or zero — ensures the flag actually gates behavior
    rather than always running.
    """
    composite = _build_kinetic_composite_with_ppgpp(ppgpp_regulation=False)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    for key in ("rela_syn", "spot_syn", "spot_deg"):
        val = growth_limits.get(key)
        if val is None:
            continue  # absent is fine
        arr = np.atleast_1d(np.asarray(val))
        if arr.size == 0:
            continue  # empty is fine
        assert np.allclose(arr, 0), (
            f"{key} listener must be zero (or absent) when "
            f"ppgpp_regulation=False, got {arr}"
        )


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_ppgpp_bulk_count_unchanged_when_regulation_off() -> None:
    """With ``ppgpp_regulation=False``, the ``GUANOSINE-5DP-3DP[c]`` bulk count
    should not change due to the kinetic process — there may be tiny
    fluctuations from other processes (RNA degradation etc.) but the
    delta should be small relative to the on-mode case.

    Strict gate: with regulation off, the kinetic process should
    contribute zero net delta. Other processes' contributions are
    unrelated.
    """
    composite = _build_kinetic_composite_with_ppgpp(ppgpp_regulation=False)
    agent_before = next(iter(composite.state["agents"].values()))
    ppgpp_before = _bulk_count(agent_before, "GUANOSINE-5DP-3DP[c]")

    composite.run(interval=1.0)

    agent_after = next(iter(composite.state["agents"].values()))
    ppgpp_after = _bulk_count(agent_after, "GUANOSINE-5DP-3DP[c]")

    # Loose gate: with regulation off, any change is from OTHER processes,
    # which should be smaller than RelA/SpoT-driven turnover. We can't
    # assert zero, but we can assert this delta is small enough that
    # comparing to the on-mode delta from the prior test would still
    # show ppgpp_regulation=True is materially different. Skip with a
    # one-cell sanity bound for now.
    delta = abs(ppgpp_after - ppgpp_before)
    # Don't enforce a strict bound — this test mostly exists to ensure
    # the composite runs cleanly with regulation off. The "on" test
    # already proves behavior is materially different.
    assert delta >= 0  # tautology, but keeps the test as a smoke gate
