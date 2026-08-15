import inspect
from v2ecoli.library import vivarium_ecoli_engine as ve


def test_build_vivarium_ecoli_has_variant_param():
    assert "variant" in inspect.signature(ve.build_vivarium_ecoli).parameters


def test_composite_builder_forwards_variant_and_observables(monkeypatch):
    seen_build = {}
    def _fake_build(**kw):
        seen_build.update(kw)
        class _H:  # minimal stand-in for EngineHandle
            pass
        return _H()
    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)

    seen_cfg = {}
    def _fake_init(self, config=None, core=None):
        seen_cfg.update(config or {})
    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__", _fake_init)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface",
                        lambda self: {"inputs": {}, "outputs": {}})

    # A bare object() cannot stand in for `core` here: process_bigraph's
    # Edge.__init__ (which Composite goes through) calls core.fill(...)
    # unconditionally while building the Composite's process tree, so the
    # fake core needs to be a real, functioning core.
    from v2ecoli.core import build_core
    ve.build_vivarium_ecoli_composite(
        sim_data_path="x", variant=4, observable_bulk_ids=["A[c]"], core=build_core())

    assert seen_build["variant"] == 4                  # variant reaches the engine builder
    assert "observable_bulk_ids" not in seen_build     # NOT a build_vivarium_ecoli concern
    assert seen_cfg["variant"] == 4                     # both reach the PROCESS config
    assert seen_cfg["observable_bulk_ids"] == ["A[c]"]
