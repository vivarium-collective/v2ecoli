"""Regression tests for config-adapter process-set key carry-through."""
import pytest
from scripts._compare.config_adapter import translate_vecoli_config


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
    # Route through _fork_registry (not a raw ``from ecoli.processes import``):
    # it saves/restores the real ecoli.* around the fork import, so it returns
    # the fork's registry order-independently WITHOUT desyncing the vivarium
    # singleton registry (which a raw sys.modules pop would not clear).
    reg = inject._fork_registry(FORK)
    assert inject.classify_process(reg.access("example-secretion")) == "vivarium_1"
    assert inject.classify_process(reg.access("bad-partitioned")) == "partitioned"

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


def test_resolve_injections_memoized(monkeypatch):
    """resolve_injections must import the fork only once per unique config."""
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 3.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    inject._RESOLVE_CACHE.clear()

    specs1 = inject.resolve_injections(FORK, cfg)
    specs2 = inject.resolve_injections(FORK, cfg)
    assert specs1 == specs2

    # Prove the cache is in use: _fork_registry must NOT be called on a
    # repeated call with an already-cached key.
    def _should_not_be_called(repo):
        raise RuntimeError("cache miss — _fork_registry called unexpectedly")

    monkeypatch.setattr(inject, "_fork_registry", _should_not_be_called)
    specs3 = inject.resolve_injections(FORK, cfg)
    assert specs3 == specs1


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

def test_apply_introduces_process_owned_store():
    """A vivarium_1 process whose topology references a store the composite does
    not yet own now INTRODUCES that store — created and materialized from the
    process's own port schema — instead of being rejected. This is what lets a
    fork subsystem carrying its own private stores (e.g. the cell-wall model's
    murein_state/wall_state) inject and run unattended. (Trade-off: a genuinely
    mis-typed store path creates an isolated store rather than erroring; the
    process simply reads/writes its own orphan store.)"""
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["nonexistent_store"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {}}
    added = inject.apply_injected_processes(cell_state, [], core, specs)
    assert "example-secretion" in added
    assert "nonexistent_store" in cell_state  # auto-introduced, not rejected


# ---------------------------------------------------------------------------
# Config initial_state honoring + schema-default backfill
# ---------------------------------------------------------------------------
def test_resolve_config_initial_state_merges_inline_and_overrides(tmp_path):
    """initial_state (inline) + initial_state_overrides (files under
    <fork>/data) merge, overrides on top."""
    fork = tmp_path / "fork"
    (fork / "data" / "overrides").mkdir(parents=True)
    (fork / "data" / "overrides" / "red.json").write_text(
        '{"murein_state": {"unincorporated_murein": 2234940}, "wall_state": {}}')
    cfg = {"initial_state": {"murein_state": {"shadow_murein": 0}},
           "initial_state_overrides": ["overrides/red"]}
    merged = inject.resolve_config_initial_state(str(fork), cfg)
    assert merged["murein_state"]["unincorporated_murein"] == 2234940
    assert merged["murein_state"]["shadow_murein"] == 0     # inline preserved
    assert merged["wall_state"] == {}


def test_apply_seeds_new_store_from_config_initial_state_not_baseline():
    """Config initial_state seeds an injected NEW store, but never clobbers a
    pre-existing v2 baseline store (e.g. the structured bulk array)."""
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["new_store"]}},
           "initial_state": {"new_store": {"seeded": 7}, "bulk": {"X": 1}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {"_real_array": True}}      # baseline store v2 owns
    inject.apply_injected_processes(cell_state, [], core, specs)
    assert cell_state["new_store"]["seeded"] == 7     # new store seeded
    assert cell_state["bulk"] == {"_real_array": True}  # baseline NOT clobbered


def test_bridge_backfills_sentinel_none_leaf():
    """The bridge presents a v1 process its declared leaves — including a
    sentinel-None default the pbg store can't persist — so reads don't KeyError."""
    from v2ecoli.library.vivarium_bridge import _backfill_schema_defaults
    schema = {"wall_state": {"lattice": {"_default": None, "_updater": "set"},
                             "rows": {"_default": 0}}}
    state = {"wall_state": {"rows": 5}}   # 'lattice' dropped by the store
    _backfill_schema_defaults(state, schema)
    assert "lattice" in state["wall_state"]
    assert state["wall_state"]["lattice"] is None     # sentinel present
    assert state["wall_state"]["rows"] == 5           # existing value untouched


# ---------------------------------------------------------------------------
# build_fork_config resolves the FORK, not the installed vEcoli
#
# A bare ``import ecoli.library.sim_data`` inside resolve_injections resolves to
# site-packages, because _fork_registry restores the installed ecoli.* as soon as
# it has the registry handle. Building a swapped process's config from the wrong
# vEcoli drops every key the fork added, and the process then falls back to its
# own class default with no error raised anywhere.
# ---------------------------------------------------------------------------

def test_build_fork_config_reads_the_fork_not_installed_vecoli():
    cfg = inject.build_fork_config(FORK, "unused.cPickle", "example-secretion")
    # Only the fixture fork's getter emits this key.
    assert cfg["fork_only_key"] == "present"
    assert cfg["rate"] == 1.0


def test_build_fork_config_raises_when_module_resolves_outside_the_fork(tmp_path):
    # A "fork" directory with no ecoli package: the import falls through to
    # whatever is installed. That must RAISE rather than return a config built
    # from the wrong source.
    with pytest.raises(inject.InjectionError, match="outside fork"):
        inject.build_fork_config(str(tmp_path), "unused.cPickle", "example-secretion")


def test_fork_resolution_guard_is_not_downgraded_to_the_default_config(tmp_path):
    # resolve_injections falls back to a default config when the fork has no
    # getter for a process. That fallback must NOT swallow the resolution guard —
    # otherwise the guard is decorative and the silent-wrong-config path returns.
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {}, "topology": {}, "time_step": 1.0,
           "fork_sim_data": "unused.cPickle"}
    monkey = inject.build_fork_config

    def _resolve_outside(fork_repo, sim_data_path, name):
        return monkey(str(tmp_path), sim_data_path, name)

    inject.resolve_injections.cache_clear() if hasattr(
        inject.resolve_injections, "cache_clear") else None
    inject.build_fork_config = _resolve_outside
    try:
        with pytest.raises(inject.InjectionError, match="outside fork"):
            inject.resolve_injections(FORK, cfg)
    finally:
        inject.build_fork_config = monkey


def test_fork_config_still_falls_back_when_the_fork_lacks_a_getter():
    # The legitimate fallback must survive: a process the fork cannot configure
    # gets the default config, not an exception.
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {}, "topology": {}, "time_step": 1.0,
           "fork_sim_data": "unused.cPickle"}
    orig = inject.build_fork_config

    def _no_getter(fork_repo, sim_data_path, name):
        raise KeyError(f"Process of name {name} is not known")

    inject.build_fork_config = _no_getter
    try:
        specs = inject.resolve_injections(FORK, cfg)
        assert specs[0]["config"] is None      # default, and no exception
    finally:
        inject.build_fork_config = orig
