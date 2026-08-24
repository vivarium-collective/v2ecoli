"""The ``bulk`` output port must be WIRED into the agent's ``bulk`` store when
observable ids are configured — otherwise process-bigraph drops the undeclared
output port and the configured observables never reach any store.

We monkeypatch the engine build + process construction (the same way
tests/test_variant_threading.py does) so no genuine EcoliSim is built, then
introspect the ``outputs`` wiring the generator (composites/vecoli.py) and the
library builder (build_vivarium_ecoli_composite) place on the agent's
``vivarium_ecoli`` node.
"""
from v2ecoli.library import vivarium_ecoli_engine as ve
from v2ecoli.core import build_core


def _patch_engine_and_process(monkeypatch, obs_ids):
    """No-op the heavy engine build + process construction; declare a ``bulk``
    output port on the fake interface exactly when ids are present (mirroring the
    real ``VivariumEcoliProcess.outputs()`` gate)."""
    def _fake_build(**kw):
        class _H:
            pass
        return _H()
    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)

    def _fake_init(self, config=None, core=None):
        self.config = dict(config or {})
    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__", _fake_init)

    outs = {"listeners": {"mass": {}}}
    if obs_ids:
        outs["bulk"] = {i: "overwrite[float]" for i in obs_ids}
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface",
                        lambda self: {"inputs": {}, "outputs": outs})


def _node_outputs_from_generator(obs_ids):
    from v2ecoli.composites.vecoli import vecoli
    doc = vecoli(core=build_core(), observable_bulk_ids=obs_ids)
    return doc["state"]["agents"]["0"]["vivarium_ecoli"]["outputs"]


def test_generator_wires_bulk_port_when_ids_present(monkeypatch):
    _patch_engine_and_process(monkeypatch, ["A[c]"])
    outputs = _node_outputs_from_generator(["A[c]"])
    assert outputs.get("listeners") == ["listeners"]
    assert outputs.get("bulk") == ["bulk"]


def test_generator_omits_bulk_port_when_no_ids(monkeypatch):
    _patch_engine_and_process(monkeypatch, [])
    outputs = _node_outputs_from_generator([])
    assert outputs.get("listeners") == ["listeners"]
    assert "bulk" not in outputs


def test_generator_threads_observables_into_process_config(monkeypatch):
    """The node must pass a declared ``observables`` (arbitrary listener leaves)
    list into the process config — the engine already emits them under the wired
    ``listeners`` port (e.g. ``peptidoglycan_shape.lysed`` for a shape/lysis
    phenotype). Mirrors observable_bulk_ids threading."""
    _patch_engine_and_process(monkeypatch, [])
    from v2ecoli.composites.vecoli import vecoli
    obs = ["peptidoglycan_shape.lysed", "mass.dry_mass"]
    doc = vecoli(core=build_core(), observables=obs)
    node = doc["state"]["agents"]["0"]["vivarium_ecoli"]
    assert node["instance"].config.get("observables") == obs
    # listeners port is always wired, so declared observables reach a store
    assert node["outputs"].get("listeners") == ["listeners"]


def test_library_builder_wires_bulk_port_when_ids_present(monkeypatch):
    _patch_engine_and_process(monkeypatch, ["A[c]"])
    composite, _info = ve.build_vivarium_ecoli_composite(
        sim_data_path="x", observable_bulk_ids=["A[c]"], core=build_core())
    node = composite.state["agents"]["0"]["vivarium_ecoli"]
    assert node["outputs"].get("bulk") == ["bulk"]
    assert node["outputs"].get("listeners") == ["listeners"]


def test_library_builder_omits_bulk_port_when_no_ids(monkeypatch):
    _patch_engine_and_process(monkeypatch, [])
    composite, _info = ve.build_vivarium_ecoli_composite(
        sim_data_path="x", observable_bulk_ids=[], core=build_core())
    node = composite.state["agents"]["0"]["vivarium_ecoli"]
    assert "bulk" not in node["outputs"]
    assert node["outputs"].get("listeners") == ["listeners"]
