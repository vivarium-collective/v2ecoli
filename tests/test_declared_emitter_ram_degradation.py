"""Guards CD2 pipeline audit finding P1-4 (§2.10): a silent
ParquetEmitter -> RAMEmitter degradation that loses ALL persisted output
without failing.

``_build_declared_emitter`` (v2ecoli/composites/_helpers.py) materialises the
generator-declared *default* emitter. When the declared address is
``ParquetEmitter`` but ``from viva_emitters import ParquetEmitter`` fails
(e.g. a stale/incomplete venv missing the parquet extras — viva-emitters is
a BASE v2ecoli dependency today, so this failure mode should be rare, but a
stale venv reproduces it exactly), the old behavior quietly built an
in-memory RAMEmitter instead and only ``warnings.warn``ed. A run that
believed it was persisting parquet to disk would run to completion, exit,
and discover every observation was gone — with no error anywhere in the
log.

The fix: that degradation now RAISES unless the caller explicitly opts in
via ``allow_ram_fallback=True`` (tests/debug only). These tests exercise
both paths, plus a positive control that a normal successful build is
unaffected.
"""

from __future__ import annotations

import builtins

import pytest

from v2ecoli.composites._helpers import _build_declared_emitter


LISTENERS_SCHEMA = {"mass": {"cell_mass": "float", "dry_mass": "float"}}


def _decl(out_dir):
    return {"address": "local:ParquetEmitter", "config": {"out_dir": str(out_dir)}}


@pytest.fixture
def broken_parquet_import(monkeypatch):
    """Make ``from viva_emitters import ParquetEmitter`` (and only that
    statement) fail with ImportError, the way it would if the [parquet]
    sub-dependencies (duckdb/polars/pyarrow) were broken/missing in a
    stale venv — without touching the real package for any other import.

    A blunter ``sys.modules['viva_emitters'] = None`` also breaks unrelated
    lazy attribute resolution inside ``process_bigraph.emitter`` (its
    ``RAMEmitter``/``SQLiteEmitter`` module-level ``__getattr__`` does its
    own ``import viva_emitters``), which would raise before
    ``_build_declared_emitter`` even reaches its ParquetEmitter branch. This
    fixture instead wraps ``builtins.__import__`` and only intercepts the
    exact ``from viva_emitters import ParquetEmitter`` form (fromlist
    contains ``"ParquetEmitter"``), leaving every other import — including
    that unrelated lazy resolution — untouched.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "viva_emitters" and fromlist and "ParquetEmitter" in fromlist:
            raise ImportError(
                "simulated: viva_emitters.ParquetEmitter unavailable "
                "(e.g. stale venv missing duckdb/polars/pyarrow)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.fast
def test_broken_parquet_import_raises_by_default(broken_parquet_import, core, tmp_path):
    """No opt-in -> a failed ParquetEmitter build raises instead of silently
    degrading to an in-memory RAMEmitter (the P1-4 silent-data-loss mode)."""
    with pytest.raises(Exception, match=r"(?i)parquet"):
        _build_declared_emitter(_decl(tmp_path), LISTENERS_SCHEMA, core)


@pytest.mark.fast
def test_broken_parquet_import_error_names_the_opt_in(broken_parquet_import, core, tmp_path):
    """The raised error is actionable: it names the opt-in knob so a
    deliberate tests/debug caller knows how to keep the old behavior."""
    with pytest.raises(Exception, match="allow_ram_fallback"):
        _build_declared_emitter(_decl(tmp_path), LISTENERS_SCHEMA, core)


@pytest.mark.fast
def test_broken_parquet_import_degrades_when_opted_in(broken_parquet_import, core, tmp_path):
    """Explicit opt-in preserves the historical degrade-to-RAM behavior."""
    from process_bigraph.emitter import RAMEmitter

    with pytest.warns(UserWarning, match=r"(?i)ram"):
        instance, _topo = _build_declared_emitter(
            _decl(tmp_path), LISTENERS_SCHEMA, core, allow_ram_fallback=True)

    assert isinstance(instance, RAMEmitter)


@pytest.mark.fast
def test_successful_parquet_build_is_unaffected(core, tmp_path):
    """Positive control: when viva_emitters IS importable, a normal declared
    ParquetEmitter build is unaffected by the new opt-in guard (default
    ``allow_ram_fallback=False`` never triggers on the success path)."""
    pytest.importorskip("viva_emitters")
    from viva_emitters import ParquetEmitter

    instance, _topo = _build_declared_emitter(_decl(tmp_path), LISTENERS_SCHEMA, core)

    assert isinstance(instance, ParquetEmitter)
