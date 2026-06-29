"""Resolve, classify, translate, and inject a fork's added processes.

Runs in the v2ecoli sim subprocess (where vivarium-core + the fork repo are
importable). The parent harness invokes the ``__main__`` below to obtain the
resolved specs as JSON for the report + early fail-fast.

Single-fork constraint: this module imports ``ecoli.processes`` once per
process class resolution; a given process lifetime must use exactly one fork
repo (the harness invokes one ``--vecoli-repo`` per run).
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from typing import Any


class InjectionError(RuntimeError):
    """A fork process cannot be injected (unsupported / unresolved)."""


@contextlib.contextmanager
def _idempotent_registration():
    """Make vivarium's ``Registry.register`` idempotent during a fork import.

    A real vEcoli fork's ``ecoli/__init__.py`` re-registers emitters/processes
    into vivarium's global singleton registries (already populated by the
    installed ``ecoli``); the stock ``register`` raises on a duplicate key whose
    value differs (a re-imported module yields new class objects). Within this
    context we skip keys that already exist and add only new ones, restoring the
    original method afterwards. If vivarium is not importable (the duck-typed
    fixture fork), this is a no-op.
    """
    try:
        from vivarium.core import registry as _vreg
    except Exception:  # noqa: BLE001 — fixture fork has no vivarium registries
        yield
        return
    orig = _vreg.Registry.register

    def _idempotent(self, key, item, alternate_keys=tuple()):
        for rk in [key, *alternate_keys]:
            if rk not in self.registry:
                self.registry[rk] = item
        self.main_keys.append(key)

    _vreg.Registry.register = _idempotent
    try:
        yield
    finally:
        _vreg.Registry.register = orig


# Cache populated by resolve_injections: (module, qualname) -> class.
# Allows _import_class to find fork classes without requiring the fork's
# ecoli.* modules to remain in sys.modules during apply_injected_processes.
_fork_class_cache: dict[tuple[str, str], type] = {}

# Memoization cache for resolve_injections: stable key -> list of spec dicts.
# Prevents re-importing the fork's ecoli.* package on every generation call
# (baseline() invokes resolve_injections once per generation; without this
# cache a real fork whose ecoli/__init__.py registers vivarium singleton
# entries would fail on duplicate registration at generation 2).
_RESOLVE_CACHE: dict[str, list] = {}


def classify_process(cls) -> str:
    """Return 'partitioned' | 'pbg_native' | 'vivarium_1' for a process class."""
    if hasattr(cls, "calculate_request") or hasattr(cls, "evolve_state"):
        return "partitioned"
    if hasattr(cls, "inputs") and hasattr(cls, "outputs"):
        return "pbg_native"
    if hasattr(cls, "ports_schema") and (
            hasattr(cls, "next_update") or hasattr(cls, "update")):
        return "vivarium_1"
    raise InjectionError(
        f"{cls.__name__}: not a recognizable process (no ports_schema/inputs).")


def _fork_registry(fork_repo: str):
    """Import the fork's ``ecoli.processes.process_registry`` and return it.

    Uses a save/restore pattern around ecoli.* in sys.modules so that:
    - The fork's ecoli.processes is loaded fresh (for registry access).
    - The installed vEcoli's ecoli.* modules are restored afterwards, preventing
      duplicate class-object registrations in vivarium singleton registries.

    Real-fork duplicate registrations: a REAL vEcoli fork's ``ecoli/__init__.py``
    re-registers emitters/processes into vivarium's GLOBAL singleton registries
    (already populated by the installed ``ecoli``), so re-importing the fork
    would raise "registry already contains an entry for ...". We make vivarium's
    ``Registry.register`` idempotent (skip keys that already exist, add new ones)
    for the duration of the fork import only, then restore it — see
    :func:`_idempotent_registration`. The fork's NEW process names (those in
    ``add_processes``) still register; names shared with the installed ecoli keep
    the installed class, which is irrelevant since we only resolve the
    added/swapped classes. The duck-typed fixture fork has no vivarium registries,
    so the context manager is a no-op there.
    """
    fork_abs = os.path.abspath(fork_repo)
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)

    # Partition current ecoli.* entries: save the real (non-fork) ones; evict all.
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        mod_file = getattr(mod, "__file__", None) or ""
        if not os.path.abspath(mod_file).startswith(fork_abs):
            saved_real[k] = mod  # keep for restore

    try:
        with _idempotent_registration():
            fork_mod = importlib.import_module("ecoli.processes")
    except Exception as exc:  # noqa: BLE001
        _restore_ecoli(saved_real, fork_repo)
        raise InjectionError(
            f"could not import 'ecoli.processes' from fork {fork_repo!r}: {exc}")

    registry = getattr(fork_mod, "process_registry", None)
    if registry is None or not hasattr(registry, "access"):
        _restore_ecoli(saved_real, fork_repo)
        raise InjectionError(
            f"fork {fork_repo!r} ecoli.processes has no process_registry.access")

    # Done with the fork's ecoli.*; restore the real vEcoli modules.
    # Class objects from the fork survive via the registry handle (and later via
    # _fork_class_cache populated in resolve_injections).
    _restore_ecoli(saved_real, fork_repo)
    return registry


def _restore_ecoli(saved_real: dict, fork_repo: str) -> None:
    """Evict fork ecoli.* from sys.modules, restore real ones, remove fork from path."""
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        del sys.modules[k]
    sys.modules.update(saved_real)
    try:
        sys.path.remove(fork_repo)
    except ValueError:
        pass


def build_fork_config(fork_repo: str, sim_data_path: str, name: str) -> dict:
    """Build a fork process's config from the FORK's own ``LoadSimData``.

    The faithful, complete config source for a converted/swapped vEcoli process:
    vEcoli's ``ecoli.library.sim_data.LoadSimData(sim_data_path).get_config_by_name``
    supplies every parameter the real process needs (where v2ecoli's reimplemented
    getter can drift). Runs in the resolve subprocess, where the fork's ``ecoli``
    package is importable. Raises if the fork has no config-getter for ``name``.
    """
    import importlib
    sim_data_mod = importlib.import_module("ecoli.library.sim_data")
    loader = sim_data_mod.LoadSimData(sim_data_path=sim_data_path)
    return dict(loader.get_config_by_name(name))


def translate_vivarium_topology(topo: dict) -> dict[str, list]:
    """Translate a vivarium-1.0 topology to process-bigraph port→store paths.

    A flat entry ``port: (a, b)`` becomes ``port: [a, b]``. A *nested* vivarium
    entry ``port: {"_path": base, sub: relpath, ...}`` wires the port to its
    ``_path`` base store (the subports live under it, matching the bridged
    process's typed sub-ports) — so a metabolism ``environment`` port declared as
    ``{"_path": ("environment",), "exchange": ("exchange",)}`` auto-wires to
    ``["environment"]`` instead of the corrupt ``["_path", "exchange"]`` a plain
    ``list()`` produced. Ports with no ``_path`` fall back to the port name.
    """
    out: dict[str, list] = {}
    for port, path in dict(topo).items():
        if isinstance(path, dict):
            base = path.get("_path", (port,))
            out[port] = list(base)
        else:
            out[port] = list(path)
    return out


def resolve_injections(fork_repo: str, config: dict) -> list[dict[str, Any]]:
    """Resolve add_processes/swap_processes -> a list of InjectionSpec dicts.

    Raises InjectionError on partitioned processes, sim_data process_configs,
    unknown names, or fork import failure.

    Results are memoized by (fork_repo, relevant config subset) so the fork's
    ecoli.* package is imported only ONCE per subprocess lifetime.  Callers
    receive a shallow copy of each cached spec dict; fail-fast InjectionErrors
    still raise normally on a cache miss (only successful results are cached).
    """
    key = json.dumps({
        "fork_repo": fork_repo,
        "add_processes": config.get("add_processes") or [],
        "swap_processes": config.get("swap_processes") or {},
        "process_configs": config.get("process_configs") or {},
        "topology": config.get("topology") or {},
        "time_step": config.get("time_step", 1.0),
    }, sort_keys=True, default=str)  # default=str: process_configs may hold
    # sim_data-derived numpy arrays (e.g. a swapped metabolism config) that are
    # not natively JSON-serializable; stringifying them keeps the memo key stable.
    if key in _RESOLVE_CACHE:
        return [dict(s) for s in _RESOLVE_CACHE[key]]

    registry = _fork_registry(fork_repo)
    interval = float(config.get("time_step", 1.0))
    process_configs = config.get("process_configs") or {}
    topologies = config.get("topology") or {}

    names = list(config.get("add_processes") or [])
    names += list((config.get("swap_processes") or {}).values())

    specs: list[dict[str, Any]] = []
    for name in names:
        try:
            cls = registry.access(name)
        except KeyError:
            raise InjectionError(f"add/swap process {name!r} not in fork registry.")
        kind = classify_process(cls)
        if kind == "partitioned":
            raise InjectionError(
                f"{name!r} is a partitioned process (calculate_request/"
                "evolve_state); not supported in v1. Extension point: wrap as "
                "PartitionedProcess (v2ecoli/steps/partition.py).")

        pcfg = process_configs.get(name, "default")
        if pcfg == "sim_data":
            raise InjectionError(
                f"{name!r}: process_configs 'sim_data' is unsupported for new "
                "processes (no ParCa entry). Provide an explicit dict or 'default'.")
        config_dict = None if pcfg in ("default", None) else dict(pcfg)
        # A 'default' config + a fork_sim_data path → auto-build the FULL, faithful
        # config from the FORK's own LoadSimData (vEcoli configures its own
        # process), instead of an empty config. Falls back to default if the fork
        # has no config-getter for this process (e.g. a brand-new add_process).
        if config_dict is None and config.get("fork_sim_data"):
            try:
                config_dict = build_fork_config(
                    fork_repo, config["fork_sim_data"], name)
            except Exception as e:  # noqa: BLE001 — not fork-configurable; use default
                print(f"[inject] fork config for {name!r} unavailable "
                      f"({type(e).__name__}); using default. {e}")
                config_dict = None

        topo = topologies.get(name)
        if topo is None:
            topo = getattr(cls, "topology", getattr(cls, "TOPOLOGY", {}))
        topo = translate_vivarium_topology(topo)

        # Cache class for apply step (survives sys.modules restore in _fork_registry).
        _fork_class_cache[(cls.__module__, cls.__qualname__)] = cls

        specs.append({
            "name": name,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "kind": kind,
            "as_step": bool(getattr(cls, "_force_step", False)),
            "config": config_dict,
            "topology": topo,
            "interval": interval,
        })
    _RESOLVE_CACHE[key] = specs
    return [dict(s) for s in specs]


def _import_class(module: str, qualname: str):
    # Check the fork class cache first (populated by resolve_injections).
    # This allows fork classes to be retrieved even after ecoli.* sys.modules
    # has been restored to the real vEcoli package.
    cached = _fork_class_cache.get((module, qualname))
    if cached is not None:
        return cached
    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def apply_injected_processes(cell_state: dict, flow_order: list, core,
                             specs: list[dict]) -> list[str]:
    """Add each resolved spec to ``cell_state`` + ``flow_order`` (in place)."""
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    from v2ecoli.composites._helpers import make_edge

    added: list[str] = []
    for spec in specs:
        cls = _import_class(spec["module"], spec["qualname"])
        if spec["kind"] == "vivarium_1":
            wrapped = wrap_vivarium_process(cls, name=spec["name"],
                                            as_step=spec["as_step"])
        else:  # pbg_native
            wrapped = cls
        core.register_link(spec["name"], wrapped)

        # Validate topology roots exist in the cell-state tree.
        for port, path in spec["topology"].items():
            root = path[0] if path else None
            if root is not None and root not in cell_state:
                raise InjectionError(
                    f"{spec['name']}: topology port {port!r} -> {path}: root "
                    f"store {root!r} not present in cell state "
                    f"(have: {sorted(cell_state)[:12]}...).")

        instance = wrapped(spec["config"] or {}, core=core)
        edge_type = "step" if spec["kind"] == "pbg_native" and spec["as_step"] \
            else ("step" if spec["as_step"] else "process")
        cell_state[spec["name"]] = make_edge(
            instance, spec["topology"], edge_type=edge_type,
            config=spec["config"] or {})
        flow_order.append(spec["name"])
        added.append(spec["name"])
    return added


def remove_processes(cell_state: dict, flow_order: list, names) -> list[str]:
    """Remove named processes/steps from a cell-state tree + flow order in place.

    The 'remove' half of a swap: a ``swap_processes`` mapping {old: new} adds the
    converted ``new`` (via :func:`apply_injected_processes`) and removes ``old``
    here; a config's ``exclude_processes`` list is removed the same way. Names not
    present are ignored (returns only the names actually removed from cell_state).
    """
    removed: list[str] = []
    for name in names:
        if name in cell_state:
            del cell_state[name]
            removed.append(name)
        while name in flow_order:
            flow_order.remove(name)
    return removed


if __name__ == "__main__":
    # argv: <fork_repo> <config_json_path>  -> prints specs JSON to stdout
    fork_repo, cfg_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    json.dump(resolve_injections(fork_repo, cfg), sys.stdout)
