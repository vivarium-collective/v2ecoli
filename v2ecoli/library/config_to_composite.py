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
