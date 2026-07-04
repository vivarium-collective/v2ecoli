"""Plasmids composite generator — baseline whole-cell model + a plasmid.

Decoupled design: the plasmid (pBR322 / ColE1) is a small, fully-specified
object — a ~4.4 kb known sequence plus literature copy-number kinetics (Brendel
& Perelson 1993). It does NOT need the whole-cell ParCa fit. So this generator
builds the STANDARD ``baseline`` document on the standard cache and then adds
the plasmid as a **purely additive layer** on top:

  * a :class:`~v2ecoli.processes.plasmid_replication.PlasmidReplication` Step,
    wired next to ``ecoli-chromosome-replication`` (plain Step, no allocator —
    the plasmid's dNTP draw is negligible vs the chromosome);
  * initial plasmid unique molecules (``full_plasmid`` / ``oriV`` /
    ``plasmid_domain`` / ``plasmid_active_replisome``);
  * the Brendel-Perelson ODE control state under
    ``process_state.plasmid_rna_control``;
  * a dedicated unique-molecule flush so the plasmid stores commit each tick;
  * minimal plasmid dividers so a dividing run is safe.

The baseline document itself is untouched, so ``baseline`` stays byte-identical
to main. All shared replication parameters come from the cache's
``ecoli-chromosome-replication`` config (see
:mod:`v2ecoli.library.plasmid_data`).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS, make_edge
from v2ecoli.composites.baseline import baseline, _derive_process_seed


_PLASMID_UNIQUE = {
    # plural port name -> singular store name (under 'unique')
    "full_plasmids": "full_plasmid",
    "oriVs": "oriV",
    "plasmid_domains": "plasmid_domain",
    "plasmid_active_replisomes": "plasmid_active_replisome",
}


def _register_plasmid_dividers() -> None:
    """Register minimal plasmid dividers WITHOUT editing baseline's division.py.

    Additive, idempotent runtime registration on the shared ``UNIQUE_DIVIDERS``
    dispatch dict. Both daughters inherit a copy of the plasmid stores (a
    ``set``-style divider — plasmid partitioning is copy-to-both). Baseline
    never has plasmid stores, so these keys are inert for plain baseline runs.
    """
    from v2ecoli.library import division

    def _divide_plasmid_set(values, unique_state):
        return values.copy(), values.copy()

    for name in _PLASMID_UNIQUE.values():
        division.UNIQUE_DIVIDERS.setdefault(name, _divide_plasmid_set)


def _add_plasmid_layer(doc: dict, core: Any, bundle: dict, seed: int) -> None:
    """Decorate a finished baseline document with the plasmid layer, in place."""
    from v2ecoli.library.config_resolver import resolve_config
    from v2ecoli.library.ecoli_step import set_current_core
    from v2ecoli.processes.plasmid_replication import (
        PlasmidReplication,
        NAME as PLASMID_NAME,
        TOPOLOGY as PLASMID_TOPOLOGY,
    )
    from v2ecoli.library.plasmid_data import (
        build_plasmid_replication_config,
        initial_plasmid_molecules,
        initial_plasmid_rna_control,
    )

    cell_state = doc["state"]["agents"]["0"]

    # --- Resolve the shared chromosome-replication config (callables realized) --
    chrom_cfg = resolve_config(bundle["configs"]["ecoli-chromosome-replication"])

    plasmid_cfg = build_plasmid_replication_config(
        chrom_cfg, seed=_derive_process_seed(seed, PLASMID_NAME)
    )

    # --- Instantiate the plasmid Step --------------------------------------
    set_current_core(core)
    try:
        plasmid_step = PlasmidReplication(plasmid_cfg)
    finally:
        set_current_core(None)

    plasmid_edge = make_edge(plasmid_step, PLASMID_TOPOLOGY, edge_type="step")

    # --- Wire it into the SAME execution layer as chromosome replication ----
    # Copy the layer-token flow wiring + priority from the chromosome edge so the
    # plasmid step is gated identically (runs in that layer, flushes at the same
    # downstream unique-update).
    chrom_edge = cell_state["ecoli-chromosome-replication"]
    for port, wire in chrom_edge.get("inputs", {}).items():
        if port.startswith("_layer_in") or port == "global_time":
            plasmid_edge.setdefault("inputs", {})[port] = wire
    for port, wire in chrom_edge.get("outputs", {}).items():
        if port.startswith("_layer_out"):
            plasmid_edge.setdefault("outputs", {})[port] = wire
    # Run just after the chromosome within the layer (deterministic ordering;
    # bulk deltas are additive so this only pins tie-breaks).
    plasmid_edge["priority"] = float(chrom_edge.get("priority", 1.0)) - 0.1

    cell_state[PLASMID_NAME] = plasmid_edge

    # --- Ensure the plasmid unique stores commit each tick ------------------
    # The baseline unique-update steps only flush the chromosome stores (the
    # cache's sim_data has no plasmid definitions). Add the plasmid stores to
    # the unique-update step that fires immediately after chromosome replication
    # so the plasmid's accumulated set/add/delete apply at the same point.
    flow_order = doc.get("flow_order", [])
    _augment_downstream_flush(cell_state, flow_order)

    # --- Insert into flow_order right after chromosome replication ----------
    if PLASMID_NAME not in flow_order:
        try:
            idx = flow_order.index("ecoli-chromosome-replication")
            flow_order.insert(idx + 1, PLASMID_NAME)
        except ValueError:
            flow_order.append(PLASMID_NAME)

    # --- Seed initial plasmid state ----------------------------------------
    unique = cell_state.setdefault("unique", {})
    templates = {
        k: unique[k]
        for k in ("full_chromosome", "oriC", "chromosome_domain", "active_replisome")
    }
    plasmid_mols = initial_plasmid_molecules(
        templates,
        dntps=list(chrom_cfg["dntps"]),
        polymerized_dntp_weights=chrom_cfg["polymerized_dntp_weights"],
        n_definitions=len(bundle.get("unique_names", [])),
    )
    for name, arr in plasmid_mols.items():
        unique[name] = arr

    process_state = cell_state.setdefault("process_state", {})
    process_state["plasmid_rna_control"] = initial_plasmid_rna_control()

    # Listener sub-store the plasmid writes.
    cell_state.setdefault("listeners", {}).setdefault("replication_data", {})

    _register_plasmid_dividers()


def _augment_downstream_flush(cell_state: dict, flow_order: list) -> None:
    """Add the plasmid stores to the first unique-update step after chromosome
    replication, so their accumulated updates flush that tick."""
    from v2ecoli.library.schema_types import UNIQUE_TYPES

    # Find the first unique_update_* step downstream of chromosome replication.
    flush_name = None
    seen_chrom = False
    for step in flow_order:
        if step == "ecoli-chromosome-replication":
            seen_chrom = True
            continue
        if seen_chrom and step.startswith("unique_update"):
            flush_name = step
            break
    if flush_name is None:
        # Fallback: any unique_update step.
        for step in flow_order:
            if step.startswith("unique_update"):
                flush_name = step
                break
    if flush_name is None or flush_name not in cell_state:
        return

    edge = cell_state[flush_name]
    instance = edge.get("instance")
    if instance is None or not hasattr(instance, "unique_topo"):
        return

    for plural, singular in _PLASMID_UNIQUE.items():
        instance.unique_topo[plural] = ("unique", singular)
        edge.setdefault("inputs", {})[plural] = ["unique", singular]
        edge.setdefault("outputs", {})[plural] = ["unique", singular]

    # Refresh the declared schemas so the engine types the new ports.
    try:
        edge["_inputs"] = instance.inputs()
        edge["_outputs"] = instance.outputs()
    except Exception:
        pass


@composite_generator(
    name="plasmids",
    description=(
        "Baseline whole-cell E. coli model with an independently-replicating "
        "plasmid (ColE1 / pBR322) under Brendel & Perelson 1993 copy-number "
        "control. Built on the STANDARD baseline cache — the plasmid is added "
        "as a purely additive layer (decoupled from the ParCa)."
    ),
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to the standard baseline ParCa cache.",
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
def plasmids(core: Any = None, *, seed: int = 0,
             cache_dir: str = "out/cache",
             bundle: dict | None = None) -> dict:
    """Build the baseline + plasmid whole-cell composite document.

    Args:
        core: bigraph-schema core. If None, one is built via build_core().
        seed: Random seed for stochastic initialisation.
        cache_dir: Path to the STANDARD baseline ParCa cache directory.
        bundle: Optional pre-loaded cache bundle (reused for both the baseline
            build and the plasmid config).

    Returns:
        Process-bigraph document dict (same shape as baseline) with the plasmid
        layer added.
    """
    if core is None:
        core = build_core()
    if bundle is None:
        bundle = load_cache_bundle(cache_dir)

    doc = baseline(core, seed=seed, cache_dir=cache_dir, bundle=bundle)
    _add_plasmid_layer(doc, core, bundle, seed)
    return doc
