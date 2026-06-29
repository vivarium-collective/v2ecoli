"""Swap semantics for fork process injection: removing the swapped-out process
and any excluded processes from the composite (the converter/add path already
exists in inject.py; this is the missing 'remove' half of a true swap)."""
import os

import pytest

from scripts._compare import inject

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")


def test_remove_processes_drops_from_state_and_flow():
    cell_state = {
        "ecoli-metabolism": {"_type": "process"},
        "keep-me": {"_type": "process"},
        "exchange_data": {"_type": "step"},
    }
    flow_order = ["ecoli-metabolism", "keep-me", "exchange_data"]

    removed = inject.remove_processes(
        cell_state, flow_order, ["ecoli-metabolism", "exchange_data"])

    assert "ecoli-metabolism" not in cell_state
    assert "exchange_data" not in cell_state
    assert "keep-me" in cell_state
    assert flow_order == ["keep-me"]
    assert set(removed) == {"ecoli-metabolism", "exchange_data"}


def test_remove_processes_ignores_absent_names():
    cell_state = {"keep-me": {"_type": "process"}}
    flow_order = ["keep-me"]

    removed = inject.remove_processes(cell_state, flow_order, ["not-here"])

    assert flow_order == ["keep-me"]
    assert removed == []


def test_resolve_builds_default_config_from_fork_sim_data(monkeypatch):
    """When fork_sim_data is provided, a process with a 'default' config gets its
    full, faithful config auto-built from the FORK's own LoadSimData (not v2's
    reimplementation), so a swapped vEcoli process is configured by vEcoli."""
    captured = {}

    def fake_build(fork_repo, sim_data_path, name):
        captured["args"] = (fork_repo, sim_data_path, name)
        return {"rate": 99.0}

    monkeypatch.setattr(inject, "build_fork_config", fake_build)
    cfg = {
        "add_processes": ["example-secretion"],
        "swap_processes": {},
        "process_configs": {},  # 'default' → should auto-build from the fork
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "fork_sim_data": "/unique/test/path/simData.cPickle",
        "time_step": 1.0,
    }
    specs = inject.resolve_injections(FORK, cfg)
    assert captured["args"][2] == "example-secretion"
    assert specs[0]["config"] == {"rate": 99.0}


def test_wrap_defer_ports_declares_any():
    """defer_ports makes the bridge declare those ports as the top type 'any', so
    a swapped process defers to the composite's existing store types (e.g. v2's
    quantity[fg] mass store) instead of imposing its unitless inferred float."""
    from v2ecoli.core import build_core
    from v2ecoli.library.ecoli_step import set_current_core
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    core = build_core()
    set_current_core(core)

    class P:
        name = "p"

        def __init__(self, parameters=None):
            self.parameters = parameters or {}

        def ports_schema(self):
            return {"a": {"_default": 0.0}, "b": {"_default": 0}}

        def next_update(self, timestep, states):
            return {}

    W = wrap_vivarium_process(P, defer_ports=["a"])
    inst = W({}, core=core)
    assert inst.inputs()["a"] == {"_type": "node"}   # defers to the store's type
    assert inst.inputs()["b"] != {"_type": "node"}
    assert inst.outputs()["a"] == {"_type": "node"}


def test_translate_vivarium_topology_resolves_nested_path():
    """vivarium nested topology ({_path: base, sub: relpath}) auto-translates to
    a process-bigraph store path (the _path base) — not the dict's keys."""
    topo = {
        "bulk": ("bulk",),
        "environment": {"_path": ("environment",), "exchange": ("exchange",)},
        "next_update_time": ("next_update_time", "metabolism"),
    }
    out = inject.translate_vivarium_topology(topo)
    assert out["bulk"] == ["bulk"]
    assert out["environment"] == ["environment"]   # _path base, NOT ['_path','exchange']
    assert out["next_update_time"] == ["next_update_time", "metabolism"]


@pytest.mark.sim
def test_baseline_swaps_and_excludes_processes():
    """A swap-only injection (no add_processes) must still convert+add the swap
    target, remove the swapped-out process, and drop exclude_processes."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    inj = {
        "fork_repo": FORK,
        "add_processes": [],
        "swap_processes": {"ecoli-mass-listener": "example-secretion"},
        "exclude_processes": ["exchange_data"],
        "process_configs": {"example-secretion": {"rate": 2.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "time_step": 1.0,
    }
    doc = baseline(core=core, seed=0, cache_dir="out/cache",
                   injected_processes=inj)
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" in cell            # swap target converted + added
    assert "ecoli-mass-listener" not in cell      # swapped-out removed
    assert "exchange_data" not in cell            # excluded removed
    assert "ecoli-mass-listener" not in doc["flow_order"]
    assert "exchange_data" not in doc["flow_order"]
