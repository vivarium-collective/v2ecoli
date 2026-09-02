"""Characterization tests for v2ecoli/bridge.py's null-emitter selection.

``EcoliWCM`` (the embedded whole-cell bridge used by colony composites) needs
its inner ``Composite`` to skip the default full-capture emitter (RAM/SQLite/
Parquet: a ~25k-molecule ``bulk`` array plus chromosome/replisome unique-node
structures, captured every tick) because the bridge reads ``composite.state``
directly and never replays the inner emitter's history. Building with the
full-state default instead produced a ~7.7 MB/sim-s per-cell RSS leak that
OOM-killed a growing colony (the #754 class; see ``EcoliWCM._build_composite``
docstring).

Historically ``bridge.py`` forced this by mutating a module-global flag
(``v2ecoli.composites._helpers._NULL_EMITTER_OVERRIDE``) directly, wrapped in
a manual save/set/restore around the ``baseline()`` call. The CD2 pipeline
audit (P2-8) flagged that pattern as fragile: any *new* embedding path that
forgets the save/restore dance silently re-inherits the full-state RAMEmitter
default. ``baseline()`` already exposes ``emitter='null'`` as a first-class,
self-contained parameter (see ``ecoli_baseline.py``'s validated ``emitter``
values: ``"parquet"``, ``"sqlite"``, ``"xarray"``, ``"null"``) that manages the
same global's full lifecycle internally, scoped to a single call — so a caller
never has an explicit restore step to forget.

These tests characterize exactly what the embedded bridge produces (so the
refactor from manual global-poking to the explicit ``emitter="null"`` argument
is provably behavior-preserving), and prove one such call cannot leak into a
concurrent/subsequent default-emitter build in the same process.
"""
import os

import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")

_NULL_EMIT_SCHEMA = {"global_time": "float"}


def _emitter_instance(doc):
    """The materialised 'emitter' step instance from a baseline() document."""
    return doc["state"]["agents"]["0"]["emitter"]["instance"]


def test_ecoli_wcm_inner_composite_uses_minimal_null_emitter():
    """The literal bridge.py integration point: EcoliWCM's inner Composite
    gets a bare global_time-only RAMEmitter, and the module-global flag is
    left exactly as it was found (False) once the build completes."""
    from process_bigraph import allocate_core
    from process_bigraph.emitter import RAMEmitter
    from v2ecoli.bridge import EcoliWCM
    from v2ecoli.composites import _helpers as _h

    _h.set_null_emitter_override(False)

    proc = EcoliWCM({"cache_dir": CACHE, "seed": 0}, core=allocate_core())
    proc._build_composite()

    inst = proc._composite.state["agents"]["0"]["emitter"]["instance"]
    assert isinstance(inst, RAMEmitter)
    assert inst.config["emit"] == _NULL_EMIT_SCHEMA
    # Scoped to the build call — never left dangling for a later caller.
    assert _h._NULL_EMITTER_OVERRIDE is False


def test_emitter_null_param_produces_minimal_ramemitter():
    """``baseline(emitter='null')`` in isolation — the mechanism bridge.py's
    refactored _build_composite() delegates to instead of poking the global
    directly."""
    from process_bigraph.emitter import RAMEmitter
    from v2ecoli.core import build_core
    from v2ecoli.composites import _helpers as _h
    from v2ecoli.composites.ecoli_baseline import baseline

    _h.set_default_emitter_decl(None)
    _h.set_parquet_emitter_override(None)
    _h.set_emitter_override(None)
    _h.set_null_emitter_override(False)

    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir=CACHE, emitter="null")
    inst = _emitter_instance(doc)
    assert isinstance(inst, RAMEmitter)
    assert inst.config["emit"] == _NULL_EMIT_SCHEMA
    assert _h._NULL_EMITTER_OVERRIDE is False


def test_null_emitter_build_does_not_leak_into_next_default_build():
    """One EcoliWCM (null-emitter) build must not affect a later default
    (parquet) build in the same process -- the module-global leak the audit
    (P2-8) raised. Proves the explicit-parameter path is leak-proof even
    though it still touches the same module global under the hood: the
    lifecycle is fully owned by baseline()'s own try/finally, not by a
    caller-managed save/restore."""
    pytest.importorskip("viva_emitters")
    from viva_emitters import ParquetEmitter
    from process_bigraph import allocate_core
    from v2ecoli.bridge import EcoliWCM
    from v2ecoli.core import build_core
    from v2ecoli.composites import _helpers as _h
    from v2ecoli.composites.ecoli_baseline import baseline

    _h.set_default_emitter_decl(None)
    _h.set_parquet_emitter_override(None)
    _h.set_emitter_override(None)
    _h.set_null_emitter_override(False)

    proc = EcoliWCM({"cache_dir": CACHE, "seed": 0}, core=allocate_core())
    proc._build_composite()
    assert _h._NULL_EMITTER_OVERRIDE is False

    # A subsequent, unrelated default-emitter build in the SAME process must
    # get the declared parquet default, not the previous call's null override.
    core = build_core()
    doc2 = baseline(core=core, seed=1, cache_dir=CACHE)
    inst2 = _emitter_instance(doc2)
    assert isinstance(inst2, ParquetEmitter)
    assert _h._NULL_EMITTER_OVERRIDE is False
