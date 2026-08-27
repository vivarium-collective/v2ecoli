"""vEcoli-style config → a process_bigraph.Composite-executable v2 document.

Standalone translator of the config's DECLARED layer (spec
docs/superpowers/specs/2026-08-26-config-to-composite-translator-design.md).
Emits address-based process nodes (config from the config's own process_configs,
wiring from topology / topology_registry) that realize once each
``local:<ClassName>`` address is registered (see register_declared_processes).
"""
from __future__ import annotations

from typing import Any

from v2ecoli.library.config_bigraph import (
    _normalize_path, _resolve_process_meta, _spatial_node, _variants_node,
)


def _node(name, topology, config, fork_dir):
    address, _desc, registry_topology = _resolve_process_meta(name, fork_dir)
    ports = dict(registry_topology)
    ports.update(topology or {})
    inputs, outputs, targets = {}, {}, set()
    for port, path in ports.items():
        norm = _normalize_path(path)
        if norm is not None:
            inputs[port] = norm            # bidirectional default (Task 3 refines)
            outputs[port] = norm
            targets.add(norm[0])
    node = {
        "_type": "process",
        "address": address,
        "config": dict(config or {}),
        "inputs": inputs,
        "outputs": outputs,
        "interval": 1.0,
    }
    return node, targets


def config_to_composite(config: dict, *, fork_dir: str = "") -> dict:
    add = list(config.get("add_processes") or [])
    swap = dict(config.get("swap_processes") or {})
    exclude = list(config.get("exclude_processes") or [])
    topology = dict(config.get("topology") or {})
    process_configs = dict(config.get("process_configs") or {})
    spatial = config.get("spatial_environment_config")
    variants = dict(config.get("variants") or {})

    state: dict[str, Any] = {}
    targets: set[str] = set()

    for name in add:
        node, t = _node(name, topology.get(name), process_configs.get(name), fork_dir)
        state[name] = node
        targets |= t
    for old, new in swap.items():
        node, t = _node(new, topology.get(new), process_configs.get(new), fork_dir)
        node.setdefault("_contract", {})["swap_replaces"] = old
        state[new] = node
        targets |= t

    for store in sorted(targets):
        state.setdefault(store, {})
    if exclude:
        state["excluded_processes"] = {e: {} for e in exclude}
    if isinstance(spatial, dict) and spatial:
        state["environment"] = _spatial_node(spatial)
    if variants:
        state["variants"] = _variants_node(variants)

    return {"schema": {}, "state": state}


def _declared_process_names(config: dict) -> list[str]:
    names = list(config.get("add_processes") or [])
    names += list((config.get("swap_processes") or {}).values())
    return names


def register_declared_processes(core, config: dict, *, fork_dir: str = "") -> list[str]:
    """Wrap each declared vivarium process via the adapter and register it under
    ``local:<ClassName>`` in ``core`` so the translated document's addresses
    resolve. Returns the list of registered class names. Best-effort per name:
    an unresolvable/unwrappable process is skipped (kept out of the return)."""
    import os, sys
    from v2ecoli.library.ecoli_step import set_current_core
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    fork = fork_dir or os.environ.get("V2E_VECOLI_DIR", "")
    if fork and fork not in sys.path:
        sys.path.insert(0, fork)
    try:
        import ecoli.processes  # noqa: F401
        from vivarium.core.registry import process_registry
    except Exception:
        return []
    # process_bigraph's realize_link instantiates the registered address
    # positionally as ``edge_class(config, core)``. EcoliProcess/EcoliStep's
    # signature is ``(parameters, config, core)`` (the vEcoli-style calling
    # convention), so that positional ``core`` lands in the ``config`` slot,
    # not ``core`` — the wrapped process's ``self.core`` would stay ``None``
    # unless the ``_CURRENT_CORE`` fallback is primed first. Deliberately
    # left set (not reset) past this function's return: the later
    # ``Composite(doc, core=core)`` call that actually realizes the
    # addresses happens outside this library's control, so the global must
    # still be primed when it runs. Same idiom as the tight-bracket use in
    # v2ecoli/composites/_helpers.py::_make_instance and ecoli_baseline.py.
    set_current_core(core)
    registered: list[str] = []
    for name in _declared_process_names(config):
        try:
            v1_cls = process_registry.access(name)
            if v1_cls is None:
                continue
            wrapped = wrap_vivarium_process(v1_cls, name=name)
            cls_name = getattr(v1_cls, "__name__", name)
            core.register_link(cls_name, wrapped)  # matches register_ecoli_core's
            registered.append(cls_name)            # Step/Process link-registration
        except Exception:
            continue
    return registered
