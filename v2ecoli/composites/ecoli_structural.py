"""baseline_parsimony — the baseline whole-cell model plus a parsimony 3D
packing step that writes snapshot packs at declared simulation times.

Wraps ``v2ecoli.composites.ecoli_baseline.ecoli_baseline`` and appends
``v2ecoli.structural.pack_step.EcoliPackStep`` as one more FINAL execution
layer on every per-agent cell state, mirroring exactly how ``baseline()``
itself appends its own ``shape_step`` final layer (see baseline.py:895-919).
"""
from __future__ import annotations

import re
from typing import Any

from viva_superpowers.composite_generator import _REGISTRY, composite_generator

from v2ecoli.composites.ecoli_baseline import baseline
from v2ecoli.composites._helpers import inject_flow_dependencies


# Pass through every parameter baseline() itself declares (seed, cache_dir,
# emitter, feature toggles, ...) so build_generator's unknown-override check
# (viva_superpowers.composite_generator.build_generator) doesn't reject them —
# this wrapper forwards **kwargs straight to baseline(core=core, **kwargs).
# ecoli_baseline() registers under the doubled id
# ``v2ecoli.composites.ecoli_baseline.ecoli_baseline`` (module slug == name=);
# the imported symbol is still named ``baseline``.
_BASELINE_PARAMS = dict(
    _REGISTRY[f"{baseline.__module__}.ecoli_baseline"].parameters)


# ---------------------------------------------------------------------------
# Flow-order reconstruction
# ---------------------------------------------------------------------------
#
# baseline() returns only a flattened per-agent 'flow_order' — its internal,
# fine-grained (parallel) execution_layers list-of-lists is a local variable,
# not part of the returned doc. Recomputing it ourselves via
# build_execution_layers(features) would require exactly reproducing the
# 'features' baseline() resolved internally (an explicit param merged with
# v2ecoli.composites.ecoli_baseline.enable_features() GLOBAL toggle state) — fragile
# to duplicate correctly.
#
# Instead we recover each step's REAL original layer index directly from the
# flow-token wiring baseline() already injected (inject_flow_dependencies,
# _helpers.py:992): a step at layer L has an input key '_layer_in_L'
# (L > 0) and/or an output key '_layer_out_L' (L < n_layers - 1). That lets
# us rebuild the exact layers list-of-lists empirically, regardless of which
# 'features' were active, and append one more final layer for the packing
# step without disturbing any pre-existing wiring (verified empirically:
# re-running inject_flow_dependencies over the reconstructed layers +
# appended layer only ADDS new keys / uniformly shifts 'priority' — it never
# collides with or reorders the existing steps).
_LAYER_IN_RE = re.compile(r"^_layer_in_(\d+)$")
_LAYER_OUT_RE = re.compile(r"^_layer_out_(\d+)$")


def _resolve_pack_out_dir(out_dir: str | None, study: str) -> str:
    """The pack step's out_dir: an explicit override wins verbatim (a caller
    that passes ``out_dir`` knows what it wants); otherwise derive the default
    from the workspace's studies root (via the shared ``viva_workspace``
    layout-aware resolver, honouring ``layout.studies``), so packs land where
    the workbench's 3D-pack viewer scans: ``<studies_root>/<study>/viz/3d``.

    The workspace root is discovered by walking up from the run's CWD; if no
    ``workspace.yaml`` is found we fall back to the conventional CWD-relative
    ``workspace/studies/<study>/viz/3d`` default (unchanged behaviour)."""
    if out_dir:
        return out_dir
    from viva_workspace import find_workspace_root, study_dir
    try:
        root = find_workspace_root()
    except FileNotFoundError:
        return f"workspace/studies/{study}/viz/3d"
    return str(study_dir(root, study) / "viz" / "3d")


def _reconstruct_execution_layers(cell_state: dict, flow_order: list[str]) -> list[list[str]]:
    """Recover baseline's per-agent execution_layers from already-injected
    flow-token wiring on ``cell_state``'s steps, in ``flow_order`` order."""
    layers: list[list[str]] = []
    for step_name in flow_order:
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict):
            continue
        idx = None
        for key in edge.get("inputs", {}):
            m = _LAYER_IN_RE.match(key)
            if m:
                idx = int(m.group(1))
                break
        if idx is None:
            for key in edge.get("outputs", {}):
                m = _LAYER_OUT_RE.match(key)
                if m:
                    idx = int(m.group(1))
                    break
        if idx is None:
            # No flow-token wiring found at all (n_layers == 1 edge case).
            idx = 0
        while len(layers) <= idx:
            layers.append([])
        layers[idx].append(step_name)
    return layers


def _append_final_step(cell_state: dict, flow_order: list[str], step_name: str) -> list[str]:
    """Append ``step_name`` as a new FINAL execution layer onto a built
    baseline() cell state, reproducing baseline's own shape_step append
    (baseline.py:895-919): reconstruct the per-agent execution_layers,
    append ``[step_name]`` as one more layer, rebuild flow_order, and call
    ``inject_flow_dependencies`` exactly as baseline.py does. Returns the
    new (extended) flow_order.
    """
    layers = _reconstruct_execution_layers(cell_state, flow_order)
    layers = layers + [[step_name]]
    new_flow_order = [step for layer in layers for step in layer]
    inject_flow_dependencies(cell_state, new_flow_order, layers=layers)
    return new_flow_order


def _register_ecoli_pack_step(core):
    """``core_extensions`` hook — register the ``EcoliPackStep`` link on the
    core the run actually builds its ``Composite`` against.

    The document build runs the generator with ``core=None``: the dashboard's
    subprocess (and ``build_generator`` in general) calls the builder without a
    core — viva_superpowers' ``CompositeSpec.to_document`` forwards that
    ``None`` straight to the builder, and the real ``build_core()`` core is
    attached only later at ``Composite(...)`` construction. So the
    ``local:EcoliPackStep`` link MUST be registered here — ``apply_core_extensions``
    runs this against the run core — NOT inside the builder body. Registering it
    in the body (guarded by ``core is not None``) meant every real run built the
    document with ``core=None``, skipped the registration AND the pack-step
    append, and silently wrote no snapshot packs.
    """
    from v2ecoli.structural.pack_step import EcoliPackStep
    core.register_link("EcoliPackStep", EcoliPackStep)
    return core


@composite_generator(
    name="ecoli_structural",
    description=(
        "baseline whole-cell E. coli model plus a parsimony 3D structural "
        "packing step (EcoliPackStep) that writes snapshot packs at declared "
        "simulation times."
    ),
    parameters={
        **_BASELINE_PARAMS,
        "study": {
            "type": "string",
            "default": "s01-birth-and-division",
            "description": "Study slug; used to derive the default pack out_dir.",
        },
        "snapshots": {
            "type": "map",
            "default": {"initial": 10.0, "pre-division": "division_time"},
            "description": "{name: fixed sim-time (s) | 'division_time'} — see EcoliPackStep.",
        },
        "top_n": {
            "type": "integer",
            "default": 40,
            "description": "Number of top-abundance ingredient species to pack.",
        },
        "scale": {
            "type": "number",
            "default": 0.3,
            "description": "Structural packing scale factor.",
        },
        "out_dir": {
            "type": "string",
            "default": None,
            "description": (
                "Explicit pack out_dir override; when unset, derived from "
                "workspace.yaml's layout.studies as "
                "'<studies_root>/<study>/viz/3d'."
            ),
        },
        "relax": {
            "type": "boolean",
            "default": False,
            "description": (
                "Relax each ingredient's structure in explicit-water MD "
                "(compacts disordered tails) before packing; cached under "
                "cache_dir/relaxed. Opt-in; adds significant first-run compute."
            ),
        },
        "relax_params": {
            "type": "object",
            "default": {},
            "description": (
                "Optional overrides for the relax MD (e.g. {equil_ps: 100.0}); "
                "see pbg_openmm.relax_in_water."
            ),
        },
        "envelope": {
            "type": "boolean",
            "default": True,
            "description": (
                "Pack into a gram-negative envelope (inner membrane + "
                "periplasm + outer membrane), routing molecules by their "
                "[compartment] tag. Default on."
            ),
        },
    },
    default_n_steps=2700,
    core_extensions=[_register_ecoli_pack_step],
)
def baseline_parsimony(
    core: Any = None,
    *,
    study: str = "s01-birth-and-division",
    snapshots: dict | None = None,
    top_n: int = 40,
    scale: float = 0.3,
    out_dir: str | None = None,
    relax: bool = False,
    relax_params: dict | None = None,
    envelope: bool = True,
    **kwargs: Any,
) -> dict:
    """Build the baseline document, then append EcoliPackStep as a final
    execution layer on every per-agent cell state."""
    doc = baseline(core=core, **kwargs)
    # Append the pack step to the built document UNCONDITIONALLY — it is pure
    # state/flow wiring and needs no core. The ``EcoliPackStep`` *link* is
    # registered separately via the ``core_extensions`` hook
    # (``_register_ecoli_pack_step``) so it lands on the run core even though
    # the standard build path (dashboard subprocess) builds this document with
    # ``core=None``. Previously the whole append was gated on ``core is not
    # None`` and returned early, which silently dropped the pack step on every
    # real run (that path always builds with ``core=None``) — no packs written.
    agents = (doc.get("state") or {}).get("agents") if isinstance(doc, dict) else None
    if not agents:
        # baseline returned a skeleton/display doc with no per-agent cell
        # states to attach the pack step to — nothing to do.
        return doc

    snaps = snapshots or {"initial": 10.0, "pre-division": "division_time"}
    # out_dir is relative to the run's CWD (the workspace root where the
    # composite is launched). When not explicitly overridden, derive it from
    # workspace.yaml's layout.studies so packs land where the workbench's 3D
    # viewer actually scans (<workspace>/<layout.studies>/<study>/viz/3d),
    # instead of a stray CWD-relative 'studies/' directory the viewer never
    # looks at.
    out_dir = _resolve_pack_out_dir(out_dir, study)
    cache_dir = kwargs.get("cache_dir") or "out/cache"

    flow_order = doc.get("flow_order") or []
    for agent_id, cell in agents.items():
        cell["pack_step"] = {
            "_type": "step",
            "address": "local:EcoliPackStep",
            "config": {
                "snapshots": snaps,
                "study": study,
                "out_dir": out_dir,
                "top_n": top_n,
                "scale": scale,
                "relax": relax,
                "cache_dir": cache_dir,
                "relax_params": relax_params or {},
                "envelope": envelope,
            },
            "inputs": {
                "bulk": ["bulk"],
                "shape": ["shape"],
                "global_time": ["global_time"],
                "full_chromosome": ["unique", "full_chromosome"],
                "active_RNAP": ["unique", "active_RNAP"],
                "active_replisome": ["unique", "active_replisome"],
                "chromosome_domain": ["unique", "chromosome_domain"],
            },
            "outputs": {"pack_status": ["pack_status"]},
        }
        new_flow_order = _append_final_step(cell, flow_order, "pack_step")

    doc["flow_order"] = new_flow_order
    return doc
