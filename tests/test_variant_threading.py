import inspect
from v2ecoli.library import vivarium_ecoli_engine as ve


def test_build_vivarium_ecoli_has_variant_param():
    assert "variant" in inspect.signature(ve.build_vivarium_ecoli).parameters


def test_composite_builder_forwards_variant(monkeypatch):
    seen = {}
    def _fake_build(**kw):
        seen.update(kw)
        class _H:  # minimal stand-in for EngineHandle
            pass
        return _H()
    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__",
                        lambda self, config=None, core=None: None)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface", lambda self: {"inputs": {}, "outputs": {}})
    # A bare object() cannot stand in for `core` here: process_bigraph's
    # Edge.__init__ (which Composite goes through) calls core.fill(...)
    # unconditionally while building the Composite's process tree, so the
    # fake core needs to be a real, functioning core.
    from v2ecoli.core import build_core
    ve.build_vivarium_ecoli_composite(
        sim_data_path="x", variant=4, observable_bulk_ids=["A[c]"], core=build_core())
    assert seen["variant"] == 4
    assert seen["observable_bulk_ids"] == ["A[c]"]
