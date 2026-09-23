"""Transform a vEcoli-style config into a loom-renderable bigraph document.

This is the fork-aware half of the workbench "config-as-bigraph" loom view
(Option 1, Phase 1). It reads the generic config vocabulary a vEcoli
``EcoliSim`` config declares —

  - ``add_processes``  (list): processes layered onto the whole cell,
  - ``swap_processes`` (dict old->new): a process replaced by another,
  - ``exclude_processes`` (list): processes dropped,
  - ``topology`` (dict proc->port->path): the per-process port wiring,
  - ``process_configs`` (dict proc->config): each process's explicit config,
  - ``spatial_environment_config`` (dict): the reaction-diffusion field,
  - ``variants`` (dict): the parameter-sweep grid,

— and emits the bigraph "state document" the vendored loom renderer already
draws (``vivarium_workbench/loom/src/convert.ts::stateToReactFlow``): a dict
keyed by node name, each ``add_processes`` entry a ``_type: "process"`` node
carrying its declared ports (``_inputs``) and, where the config's ``topology``
declares a flat store path, an ``inputs`` wire so the loom draws the edge.

Phase 1 is deliberately honest about what it can show without instantiating the
simulator: every declared port renders (from ``_inputs``), but only ports whose
``topology`` value is a simple store path are *wired*; deeply-nested sub-port
topologies render as un-wired ports (Phase 2 resolves those). Nodes are badged
``_draft: True`` — this is the config's declared structure, not a built cell.

``fork_dir`` (or ``$V2E_VECOLI_DIR``) is used only to enrich each node with the
real registered class address + docstring; the graph shape needs no fork.
"""

from __future__ import annotations

from typing import Any

# Keys that carry per-process detail, not their own graph node.
_ANNOTATION_KEYS = (
    "add_processes", "swap_processes", "exclude_processes",
    "topology", "process_configs", "spatial_environment_config",
    "variants", "flow",
)


def _normalize_path(path: Any) -> "list[str] | None":
    """Turn a config ``topology`` value into a flat root-level store path.

    The config authors paths relative to the process's place under
    ``agents/<id>/...`` (leading ``..`` walk-ups); at the root of a config
    document those walk-ups have nothing to climb, so we strip them and keep
    the remaining tail as a root store path. Returns None for a value that is
    NOT a flat list of path segments (e.g. a nested sub-port dict) — those are
    left un-wired in Phase 1 rather than mis-wired.
    """
    if not isinstance(path, (list, tuple)) or not path:
        return None
    tail = [str(seg) for seg in path if seg not in ("..", None, "")]
    # A ``["null"]`` sentinel or an all-``..`` path leaves nothing to wire.
    if not tail or tail == ["null"]:
        return None
    return tail


def _fork_registries(fork_dir: str):
    """Import the fork's ``(process_registry, topology_registry)`` once.

    ``fork_dir``/``$V2E_VECOLI_DIR`` selects the vEcoli checkout. Returns
    ``(None, None)`` when no fork is available or the import fails — every
    caller degrades gracefully, so the graph shape is identical fork-less.
    """
    try:
        import os
        import sys

        fork = fork_dir or os.environ.get("V2E_VECOLI_DIR", "")
        if fork and fork not in sys.path:
            sys.path.insert(0, fork)
        import ecoli.processes  # noqa: F401 — registers the process classes
        from ecoli.processes.registries import topology_registry
        from vivarium.core.registry import process_registry

        return process_registry, topology_registry
    except Exception:  # noqa: BLE001 — enrichment is optional, never fatal
        return None, None


def _resolve_process_meta(name: str, fork_dir: str) -> "tuple[str, str, dict]":
    """Best-effort (address, description, registry_topology) from the fork.

    ``registry_topology`` is the fork ``topology_registry``'s port->path map for
    this process (``{}`` when absent) — the fallback ports for a process whose
    wiring the *config* does not itself declare (e.g. pg-shape / pg-maturation,
    whose topology lives in the registry, not the antibiotic config).
    """
    address = f"local:{name}"
    description = ""
    registry_topology: dict = {}
    process_registry, topology_registry = _fork_registries(fork_dir)
    if process_registry is not None:
        try:
            cls = process_registry.access(name)
            if cls is not None:
                address = f"local:{getattr(cls, '__name__', name)}"
                doc = (cls.__doc__ or "").strip()
                if doc:
                    description = doc.splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            pass
    if topology_registry is not None:
        try:
            topo = topology_registry.access(name)
            if isinstance(topo, dict):
                registry_topology = topo
        except Exception:  # noqa: BLE001
            pass
    return address, description, registry_topology


def _process_node(
    name: str,
    topology: "dict | None",
    config: "dict | None",
    fork_dir: str,
) -> "tuple[dict, set[str]]":
    """Build one ``_type: process`` node plus the set of root stores it wires to.

    ``_inputs`` lists every declared port (so all ports render); ``inputs``
    carries a wire only for ports whose ``topology`` is a flat store path.
    """
    address, description, registry_topology = _resolve_process_meta(name, fork_dir)
    # Config-declared wiring is authoritative; the fork's topology_registry
    # fills in ports for a process the config leaves un-wired (pg-shape etc.).
    ports = dict(registry_topology)
    ports.update(topology or {})

    inputs_schema: dict[str, str] = {}
    wires: dict[str, list] = {}
    targets: set[str] = set()
    for port, path in ports.items():
        inputs_schema[port] = "any"
        norm = _normalize_path(path)
        if norm is not None:
            wires[port] = norm
            targets.add(norm[0])

    node: dict[str, Any] = {
        "_type": "process",
        "address": address,
        "config": dict(config or {}),
        "_inputs": inputs_schema,
        "_outputs": {},
        "_draft": True,
    }
    if description:
        node["description"] = description
    if wires:
        node["inputs"] = wires
    return node, targets


def _spatial_node(spatial: dict) -> dict:
    """A store node summarizing the reaction-diffusion field (not a process)."""
    node: dict[str, Any] = {}
    rd = spatial.get("reaction_diffusion") if isinstance(spatial, dict) else None
    if isinstance(rd, dict):
        mols = rd.get("molecules") or rd.get("gradient", {}).get("molecules")
        if isinstance(mols, (list, dict)):
            for m in (mols.keys() if isinstance(mols, dict) else mols):
                node[str(m)] = {}
    # Keep at least one child so it renders as a labeled container.
    if not node:
        node["field"] = {}
    return node


def _variants_node(variants: dict) -> dict:
    """A store node listing the sweep-grid variant name(s)."""
    return {str(k): {} for k in variants} or {"grid": {}}


def config_to_document(config: dict, *, fork_dir: str = "") -> dict:
    """Transform a (resolved) vEcoli config dict into a loom state document.

    Returns ``{"state": {...}, "summary": {...}}``. ``state`` is the bigraph
    document the loom renders directly; ``summary`` is a small non-rendered
    digest (counts) handy for a caller/test.
    """
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
        node, node_targets = _process_node(
            name, topology.get(name), process_configs.get(name), fork_dir)
        state[name] = node
        targets |= node_targets

    for old, new in swap.items():
        node, node_targets = _process_node(
            new, topology.get(new), process_configs.get(new), fork_dir)
        node["description"] = (
            f"SWAP — replaces '{old}'. " + node.get("description", "")).strip()
        node.setdefault("_contract", {})["swap_replaces"] = old
        state[new] = node
        targets |= node_targets

    # Root store nodes for every wire target, so edges have endpoints. Never
    # clobber a process node that happens to share a store's name.
    for store in sorted(targets):
        if store not in state:
            state[store] = {}

    if exclude:
        state["excluded_processes"] = {e: {} for e in exclude}
    if isinstance(spatial, dict) and spatial:
        state["environment"] = _spatial_node(spatial)
    if variants:
        state["variants"] = _variants_node(variants)

    summary = {
        "added": add,
        "swapped": swap,
        "excluded": exclude,
        "n_process_nodes": len(add) + len(swap),
        "n_wired_ports": sum(
            len(n.get("inputs", {})) for n in state.values()
            if isinstance(n, dict) and n.get("_type") == "process"),
        "has_spatial": bool(spatial),
        "n_variants": len(variants),
    }
    return {"state": state, "summary": summary}
