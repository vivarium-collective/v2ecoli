"""The baseline generator ships a ParquetEmitter as its default sink.

Exercises the pbg_superpowers ``@composite_generator(emitters=[...])``
convention end-to-end: the baseline generator declares a ParquetEmitter
default, and the 'emitter' step materialises it when no external override is
set — while external overrides still win. See
``v2ecoli.composites._helpers.set_default_emitter_decl`` and
``pbg_superpowers.composite_generator.emitter_defaults``.
"""
import os
import pytest

pytest.importorskip("pbg_emitters")

from pbg_emitters import ParquetEmitter  # noqa: E402
from process_bigraph.emitter import SQLiteEmitter  # noqa: E402

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def _emitter_instance(doc):
    """The materialised 'emitter' step instance from a baseline document."""
    return doc["state"]["agents"]["0"]["emitter"]["instance"]


def test_baseline_declares_parquet_default():
    """The generator advertises a ParquetEmitter via emitter_defaults()."""
    from pbg_superpowers.composite_generator import emitter_defaults
    from v2ecoli.composites.baseline import baseline

    decls = emitter_defaults(baseline)
    assert decls and decls[0]["address"] == "local:ParquetEmitter"


def test_baseline_uses_parquet_when_no_override(tmp_path):
    """No external override -> the emitter step is a ParquetEmitter."""
    from v2ecoli.core import build_core
    from v2ecoli.composites import _helpers as _h
    from v2ecoli.composites.baseline import baseline

    # Sink to a tmp dir so the test never writes into the repo's .pbg.
    _h.set_default_emitter_decl(None)  # clean slate
    _h.set_parquet_emitter_override(None)
    _h.set_emitter_override(None)

    core = build_core()
    # No external override and (typically) no workspace.yaml above the repo,
    # so out_dir resolves to a relative default; we only build the doc (never
    # run it), so no parquet files are written.
    doc = baseline(core=core, seed=0, cache_dir=CACHE)
    inst = _emitter_instance(doc)
    assert isinstance(inst, ParquetEmitter)
    # The decl is published only for the duration of the build, then cleared.
    assert _h._DEFAULT_EMITTER_DECL is None


def test_external_sqlite_override_beats_declared_parquet_default(tmp_path):
    """An external sqlite override wins over the generator's parquet default."""
    from v2ecoli.core import build_core
    from v2ecoli.composites import _helpers as _h
    from v2ecoli.composites.baseline import baseline

    core = build_core()
    _h.set_parquet_emitter_override(None)
    _h.set_emitter_override({
        "file_path": str(tmp_path),
        "db_file": "history.sqlite",
        "simulation_id": "test-default-emitter",
    })
    try:
        doc = baseline(core=core, seed=0, cache_dir=CACHE)
        inst = _emitter_instance(doc)
        assert isinstance(inst, SQLiteEmitter)
    finally:
        _h.set_emitter_override(None)
    assert _h._DEFAULT_EMITTER_DECL is None
