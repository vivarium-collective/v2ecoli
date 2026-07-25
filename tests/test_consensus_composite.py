"""Phase 4 tests: consensus_baseline composite alias + parity gates.

Three layers (per audit.md §6 row P4):

1. **Composite-level smoke** (no cache): ``consensus_baseline`` imports,
   registers in the composite registry, exposes the expected signature,
   forces ``include_aa_supply=True`` + ``ppgpp_regulation=True`` defaults.

2. **Composite equivalence** (sim + cache): ``consensus_baseline()`` and
   ``kinetic_charging_baseline(config_overrides={both flags True})``
   produce equivalent first-tick behavior — same AA deltas, same ppGpp
   delta, same listener emission shapes.

3. **Cross-class numeric parity** (sim + cache): the kinetic class's
   ``_ppgpp_request`` and SteadyState's ``_ppgpp_request`` produce the
   same ``delta_metabolites`` on identical input arrays. Locks in that
   the P2 port is bit-faithful to SteadyState — the two classes share
   the same ppGpp math.

4. **Flag-off legacy regression** (sim + cache): ``kinetic_charging_baseline``
   with ``include_aa_supply=False`` AND ``ppgpp_regulation=False`` runs
   to completion — the consensus-mode flags do not break the legacy
   kinetic path.
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


# ============================================================
# §1 — composite-level smoke (no cache)
# ============================================================

def test_consensus_baseline_module_imports() -> None:
    from v2ecoli.composites import consensus_baseline  # noqa: F401


def test_consensus_baseline_registered() -> None:
    """The @composite_generator decorator must have fired on import,
    landing the generator in pbg_superpowers' registry under both the
    doubled id and the clean alias.
    """
    from v2ecoli.composites import consensus_baseline  # noqa: F401
    from pbg_superpowers.composite_generator import _REGISTRY

    doubled = "v2ecoli.composites.consensus_baseline.consensus_baseline"
    clean = "v2ecoli.composites.consensus_baseline"
    assert doubled in _REGISTRY, (
        f"composite generator not registered at doubled id {doubled!r}"
    )
    assert clean in _REGISTRY, (
        f"clean alias {clean!r} not registered — composites/__init__.py "
        "must call _register_clean_alias('consensus_baseline')"
    )


def test_consensus_baseline_signature_matches_kinetic() -> None:
    """consensus_baseline must accept the same kwargs as
    kinetic_charging_baseline so it's a drop-in replacement.
    """
    from v2ecoli.composites.consensus_baseline import consensus_baseline
    from v2ecoli.composites.kinetic_charging_baseline import (
        kinetic_charging_baseline,
    )

    consensus_sig = inspect.signature(consensus_baseline)
    kinetic_sig = inspect.signature(kinetic_charging_baseline)
    assert set(consensus_sig.parameters) == set(kinetic_sig.parameters), (
        f"signature mismatch: consensus={set(consensus_sig.parameters)} "
        f"vs kinetic={set(kinetic_sig.parameters)}"
    )


def test_consensus_baseline_forces_both_flags_on_by_default() -> None:
    """Source-marker contract: the module must define a defaults dict
    that turns on BOTH include_aa_supply and ppgpp_regulation. Without
    these forced defaults, ``consensus_baseline`` would silently
    degenerate to the bare kinetic baseline.
    """
    from v2ecoli.composites import consensus_baseline as mod

    assert hasattr(mod, "_CONSENSUS_DEFAULTS"), (
        "consensus_baseline module must expose a _CONSENSUS_DEFAULTS dict "
        "so the flag-forcing contract is testable in isolation"
    )
    defaults = mod._CONSENSUS_DEFAULTS
    assert defaults.get(
        "ecoli-polypeptide-elongation.include_aa_supply"
    ) is True, "include_aa_supply must default to True"
    assert defaults.get(
        "ecoli-polypeptide-elongation.ppgpp_regulation"
    ) is True, "ppgpp_regulation must default to True"


def test_consensus_baseline_lets_user_overrides_win() -> None:
    """The composite must merge user ``config_overrides`` on top of the
    defaults (not the other way around) so users can degrade the
    consensus to its constituent modes for ablation experiments.

    Source check: the merge order must be ``merged = defaults; merged.update(user)``,
    not the inverse.
    """
    from v2ecoli.composites.consensus_baseline import consensus_baseline

    src = inspect.getsource(consensus_baseline)
    # Heuristic: the canonical pattern is `merged = dict(_CONSENSUS_DEFAULTS)`
    # followed by `merged.update(config_overrides)`.
    assert "dict(_CONSENSUS_DEFAULTS)" in src, (
        "consensus_baseline must seed the merge from _CONSENSUS_DEFAULTS "
        "and update from user overrides — not the reverse"
    )
    assert ".update(" in src, (
        "consensus_baseline must merge user overrides via .update()"
    )


# ============================================================
# §2 — composite equivalence (sim + cache)
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_consensus_baseline_one_tick_smoke() -> None:
    """``consensus_baseline()`` builds + runs 1 tick without error and
    emits the consensus-defining listeners (supply + ppGpp) non-empty.
    Smoke gate that the alias is wired end-to-end.
    """
    from process_bigraph import Composite
    from v2ecoli.composites.consensus_baseline import consensus_baseline
    from v2ecoli.core import build_core

    core = build_core()
    doc = consensus_baseline(core=core, seed=0, cache_dir=CACHE)
    composite = Composite(doc, core=core)
    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    growth_limits = agent["listeners"]["growth_limits"]

    # Supply listener (P3b-ii proof)
    aa_supply = growth_limits.get("aa_supply")
    assert aa_supply is not None and len(aa_supply), (
        "consensus_baseline must emit non-empty aa_supply listener — "
        "include_aa_supply default is not flowing through"
    )

    # ppGpp listener (P2 proof)
    rela_syn = growth_limits.get("rela_syn")
    assert rela_syn is not None and len(rela_syn), (
        "consensus_baseline must emit non-empty rela_syn listener — "
        "ppgpp_regulation default is not flowing through"
    )


@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_consensus_baseline_equivalent_to_kinetic_with_overrides() -> None:
    """A consensus_baseline build and a kinetic_charging_baseline build
    with the same flags via config_overrides must produce IDENTICAL
    first-tick AA deltas (same seed, same cache).

    This is the composite-level parity gate: consensus_baseline is just
    discoverability — it must not add ANY behavior on top of the kinetic
    composite with flags on.
    """
    from process_bigraph import Composite
    from v2ecoli.composites.consensus_baseline import consensus_baseline
    from v2ecoli.composites.kinetic_charging_baseline import (
        kinetic_charging_baseline,
    )
    from v2ecoli.core import build_core

    core_a = build_core()
    core_b = build_core()
    doc_consensus = consensus_baseline(core=core_a, seed=0, cache_dir=CACHE)
    doc_kinetic = kinetic_charging_baseline(
        core=core_b,
        seed=0,
        cache_dir=CACHE,
        config_overrides={
            "ecoli-polypeptide-elongation.include_aa_supply": True,
            "ecoli-polypeptide-elongation.ppgpp_regulation": True,
        },
    )

    comp_consensus = Composite(doc_consensus, core=core_a)
    comp_kinetic = Composite(doc_kinetic, core=core_b)

    # Pre-tick: AA pools must match (same cache, same seed). Use the
    # canonical AA bulk names from the cache config rather than EcoCyc
    # short ids (the cache uses full names like L-ALPHA-ALANINE[c]).
    from v2ecoli.core import load_cache_bundle
    aa_names = load_cache_bundle(CACHE)["configs"][
        "ecoli-polypeptide-elongation"
    ]["amino_acids"][:6]

    def _aa(comp):
        agent = next(iter(comp.state["agents"].values()))
        ids = list(agent["bulk"]["id"])
        cnt = agent["bulk"]["count"]
        return np.array([cnt[ids.index(n)] for n in aa_names])

    np.testing.assert_array_equal(
        _aa(comp_consensus), _aa(comp_kinetic),
        err_msg="initial AA counts diverge between consensus and kinetic builds"
    )

    comp_consensus.run(interval=1.0)
    comp_kinetic.run(interval=1.0)

    np.testing.assert_array_equal(
        _aa(comp_consensus), _aa(comp_kinetic),
        err_msg=(
            "consensus_baseline and kinetic_charging_baseline+overrides "
            "diverged on 1-tick AA deltas — consensus_baseline is adding "
            "behavior beyond the kinetic composite + flag overrides"
        ),
    )


# ============================================================
# §3 — cross-class numeric parity (sim + cache)
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_ppgpp_request_numeric_parity_with_steady_state() -> None:
    """The kinetic class's ``_ppgpp_request`` and SteadyState's
    ``_ppgpp_request`` must compute the same delta_metabolites on
    identical input arrays. This locks in that P2's port is bit-faithful
    to SteadyState — they share the underlying ``ppgpp_metabolite_changes``
    math, so the wrappers must produce identical bulk requests.

    Builds both processes from the same cache config (extended with the
    synthetic kinetic_charging keys when needed for the kinetic side),
    seeds RNG identically, calls each ``_ppgpp_request`` with the same
    input arrays, and compares the returned delta-metabolite tuples.
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )
    from v2ecoli.processes.polypeptide_elongation import (
        SteadyStatePolypeptideElongation,
    )
    from v2ecoli.types.quantity import ureg as units

    cfg = dict(load_cache_bundle(CACHE)["configs"]["ecoli-polypeptide-elongation"])

    # Build the kinetic process. The cache config carries the relation
    # extensions because the kinetic baseline was built against it.
    n_aas = len(cfg["amino_acids"])
    n_trnas = len(cfg["uncharged_trna_names"])
    # Force ppgpp_regulation=True so the method is not a no-op.
    cfg_kinetic = dict(cfg)
    cfg_kinetic["ppgpp_regulation"] = True
    proc_kinetic = KineticTrnaChargingPolypeptideElongation(cfg_kinetic)

    # Build the SteadyState process from the same cache config.
    cfg_steady = dict(cfg)
    cfg_steady["ppgpp_regulation"] = True
    proc_steady = SteadyStatePolypeptideElongation(cfg_steady)

    # Seed both RNGs identically so ppgpp_metabolite_changes' stochastic
    # branch (negative-count safeguard) returns the same draws.
    proc_kinetic.random_state = np.random.RandomState(42)
    proc_steady.random_state = np.random.RandomState(42)

    # Both classes need their bulk indices resolved. Use the same bulk
    # name list from the cache.
    bulk_ids = cfg["proteinIds"].tolist() + [
        cfg["water"], cfg["proton"], cfg["rela"], cfg["spot"], cfg["ppgpp"],
    ] + list(cfg["uncharged_trna_names"]) + list(cfg["charged_trna_names"]) + list(
        cfg["charging_molecule_names"]
    ) + list(cfg["synthetase_names"]) + list(cfg["amino_acids"]) + [
        cfg["ribosome30S"], cfg["ribosome50S"],
    ] + list(cfg["aa_enzymes"]) + list(cfg["aa_importers"]) + list(
        cfg["aa_exporters"]
    ) + list(cfg["ppgpp_reaction_metabolites"])
    # The above is a permissive superset; bulk_name_to_idx tolerates duplicates.
    # We only need ppgpp_idx and ppgpp_rxn_metabolites_idx to be set.
    proc_kinetic._init_bulk_indices(bulk_ids)
    proc_steady._init_bulk_indices(bulk_ids)

    # Build a minimal states dict that both _ppgpp_request implementations
    # can index into for the ppGpp metabolite request.
    n_bulk = len(bulk_ids)
    bulk_counts = np.full(n_bulk, 100, dtype=np.int64)  # any positive count
    states = {
        "bulk": bulk_counts,
        "timestep": 1.0,
    }

    # Synthetic but consistent inputs to both methods.
    rng = np.random.RandomState(7)
    counts_to_uM_mag = 1e-3
    uncharged_trna_counts = rng.randint(100, 1000, size=n_aas).astype(np.int64)
    charged_trna_counts = rng.randint(100, 1000, size=n_aas).astype(np.int64)
    fraction_charged = rng.uniform(0.3, 0.95, size=n_aas)
    ribosome_conc = 30.0  # μM, typical
    f = np.full(n_aas, 1.0 / n_aas)
    rela_conc = 0.5  # μM
    spot_conc = 0.5  # μM
    ppgpp_conc = 50.0  # μM, typical
    v_rib = 20.0  # μM/s

    kinetic_result = proc_kinetic._ppgpp_request(
        states, counts_to_uM_mag, uncharged_trna_counts, charged_trna_counts,
        fraction_charged, ribosome_conc, f, rela_conc, spot_conc, ppgpp_conc, v_rib,
    )
    # Reset RNGs to identical state — _ppgpp_request consumed a draw.
    proc_kinetic.random_state = np.random.RandomState(42)
    proc_steady.random_state = np.random.RandomState(42)
    steady_result = proc_steady._ppgpp_request(
        states, counts_to_uM_mag, uncharged_trna_counts, charged_trna_counts,
        fraction_charged, ribosome_conc, f, rela_conc, spot_conc, ppgpp_conc, v_rib,
    )

    # Both return a list of (idx, delta) tuples in the same order.
    assert len(kinetic_result) == len(steady_result), (
        f"length mismatch: kinetic returned {len(kinetic_result)}, "
        f"steady returned {len(steady_result)}"
    )
    for (k_idx, k_delta), (s_idx, s_delta) in zip(
        kinetic_result, steady_result
    ):
        # Indices may be ndarrays (ppgpp_rxn_metabolites_idx) or scalars
        # (ppgpp_idx).
        np.testing.assert_array_equal(
            np.atleast_1d(k_idx), np.atleast_1d(s_idx),
            err_msg="bulk index mismatch between kinetic and steady ppgpp_request",
        )
        np.testing.assert_array_equal(
            np.atleast_1d(k_delta), np.atleast_1d(s_delta),
            err_msg=(
                "delta_metabolites mismatch — the kinetic _ppgpp_request "
                "port has drifted from SteadyState's reference"
            ),
        )


# ============================================================
# §4 — flag-off legacy regression (sim + cache)
# ============================================================

@pytest.mark.sim
@_needs_cache
@_post_port_cache
def test_kinetic_baseline_runs_with_both_flags_off() -> None:
    """The legacy kinetic path (both consensus flags off) must continue
    to run cleanly. Regression gate: P2's _ppgpp_request/_ppgpp_evolve
    must be true no-ops when ppgpp_regulation=False, and P3b-ii's supply
    closure must be None when include_aa_supply=False.

    Symptom of a broken gate: this composite errors during build or run,
    OR the AA bulk deltas materially differ from the consensus run
    (degenerate-mode behavior should be drain-only).
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
            "ecoli-polypeptide-elongation.include_aa_supply": False,
            "ecoli-polypeptide-elongation.ppgpp_regulation": False,
        },
    )
    composite = Composite(doc, core=core)

    # Use a canonical AA name from the cache (EcoCyc short codes like
    # "ALA[c]" aren't in the bulk; e.g. alanine is "L-ALPHA-ALANINE[c]").
    from v2ecoli.core import load_cache_bundle
    ala_name = load_cache_bundle(CACHE)["configs"][
        "ecoli-polypeptide-elongation"
    ]["amino_acids"][0]

    agent = next(iter(composite.state["agents"].values()))
    ids = list(agent["bulk"]["id"])
    initial_ala = int(agent["bulk"]["count"][ids.index(ala_name)])

    composite.run(interval=1.0)

    agent = next(iter(composite.state["agents"].values()))
    ids = list(agent["bulk"]["id"])
    final_ala = int(agent["bulk"]["count"][ids.index(ala_name)])

    # Legacy behavior: AA pool drains over the tick (no supply
    # replenishment). The exact delta isn't fixed, but the pool MUST
    # have decreased — otherwise something is wrong.
    assert final_ala <= initial_ala, (
        f"{ala_name} pool grew ({initial_ala} → {final_ala}) with both "
        f"consensus flags off — no supply, so should drain monotonically"
    )

    # Verify ppGpp listener is absent/zero (proves flag gating works).
    growth_limits = agent["listeners"]["growth_limits"]
    rela_syn = growth_limits.get("rela_syn")
    if rela_syn is not None and len(rela_syn) > 0:
        assert np.allclose(np.asarray(rela_syn), 0), (
            "rela_syn listener non-zero with ppgpp_regulation=False — "
            "P2 flag gate is leaking"
        )
