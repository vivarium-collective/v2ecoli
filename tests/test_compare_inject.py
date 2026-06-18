"""Regression tests for config-adapter process-set key carry-through."""
import sys
import pytest
from scripts._compare.config_adapter import translate_vecoli_config


@pytest.fixture(autouse=True)
def _isolate_ecoli_modules():
    """Save and remove ecoli entries from sys.modules before each test, restore after.

    Prevents earlier tests that import the real ``ecoli`` package from polluting
    sys.modules when this file's tests do ``sys.path.insert(0, FORK)`` and
    re-import ``ecoli.processes`` — the cached real package would otherwise be
    returned, hiding the fixture fork's ExampleSecretion / BadPartitioned.
    """
    saved = {k: v for k, v in sys.modules.items()
             if k == "ecoli" or k.startswith("ecoli.")}
    for k in saved:
        sys.modules.pop(k)
    yield
    # Remove anything the test added, then restore the pre-test state.
    for k in list(sys.modules.keys()):
        if k == "ecoli" or k.startswith("ecoli."):
            sys.modules.pop(k)
    sys.modules.update(saved)


def test_translate_preserves_process_set_keys():
    vecoli = {
        "experiment_id": "x", "generations": 2,
        "add_processes": ["example-secretion"],
        "swap_processes": {}, "exclude_processes": [],
        "process_configs": {"example-secretion": {"rate": 2.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "emitter": "parquet",            # vEcoli-only -> dropped
    }
    v2 = translate_vecoli_config(vecoli)
    assert v2["add_processes"] == ["example-secretion"]
    assert v2["process_configs"]["example-secretion"] == {"rate": 2.0}
    assert v2["topology"]["example-secretion"] == {"counts": ["bulk"]}
    assert "emitter" not in v2                      # still dropped
    assert v2["_dropped_vecoli_keys"]["emitter"] == "parquet"


# ---------------------------------------------------------------------------
# Task 2: classify_process + resolve_injections
# ---------------------------------------------------------------------------
import os
import pytest
from scripts._compare import inject

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")

def test_classify_vivarium_and_partitioned():
    import sys; sys.path.insert(0, FORK)
    from ecoli.processes import ExampleSecretion, BadPartitioned
    assert inject.classify_process(ExampleSecretion) == "vivarium_1"
    assert inject.classify_process(BadPartitioned) == "partitioned"

def test_resolve_injections_builds_spec():
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 3.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    assert len(specs) == 1
    s = specs[0]
    assert s["name"] == "example-secretion"
    assert s["kind"] == "vivarium_1"
    assert s["config"] == {"rate": 3.0}
    assert s["topology"] == {"counts": ["bulk"]}
    assert s["qualname"] == "ExampleSecretion"

def test_resolve_rejects_partitioned():
    cfg = {"add_processes": ["bad-partitioned"], "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="partitioned"):
        inject.resolve_injections(FORK, cfg)

def test_resolve_rejects_sim_data_config():
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": "sim_data"},
           "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="sim_data"):
        inject.resolve_injections(FORK, cfg)

def test_resolve_rejects_unknown_name():
    cfg = {"add_processes": ["no-such-process"], "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="not in fork registry"):
        inject.resolve_injections(FORK, cfg)


# ---------------------------------------------------------------------------
# Task 3: apply_injected_processes
# ---------------------------------------------------------------------------
def test_apply_injects_edge_and_flow_order():
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": {"rate": 2.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {}}      # a 'bulk' store exists
    flow_order = ["ecoli-metabolism"]
    added = inject.apply_injected_processes(cell_state, flow_order, core, specs)
    assert added == ["example-secretion"]
    assert "example-secretion" in cell_state
    edge = cell_state["example-secretion"]
    assert edge["_type"] in ("process", "step")
    assert edge["inputs"]["counts"] == ["bulk"]
    assert flow_order[-1] == "example-secretion"

def test_apply_rejects_missing_store_path():
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["nonexistent_store"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    with pytest.raises(inject.InjectionError, match="nonexistent_store"):
        inject.apply_injected_processes({"bulk": {}}, [], core, specs)
