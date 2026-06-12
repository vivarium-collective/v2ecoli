"""Millard FBA-bridge harness composite — the *integration climax* of the
Millard FBA-bridge flux-pin feature.

Unlike ``millard_pdmp_baseline`` (which REMOVES the WCM's tFBA ``Metabolism``
and replaces it with the Millard ODE stack), this harness keeps the REAL WCM
``ecoli-metabolism`` and uses the Millard 2017 kinetic ODE as a flux *source*
that mechanistically pins the central-carbon FBA reactions.

Wiring strategy
---------------
Start from the full ``baseline`` build (all ~55 WCM processes, including the
real ``ecoli-metabolism`` tFBA step). Insert two edges in the layer slot
immediately BEFORE ``ecoli-metabolism`` so they run first each tick:

  1. ``millard-pdmp-metabolism`` (``MillardPDMPMetabolism``): the COPASI/basico
     Millard 2017 central-carbon ODE. We use ONLY its ``central_fluxes`` output
     ({millard_rxn -> mM/s}); its bulk writeback is intentionally routed to a
     throwaway sink store so it never perturbs the WCM bulk (the real
     ``ecoli-metabolism`` is the sole metabolic bulk driver).
  2. ``fba-flux-coupler`` (``FBAFluxCoupler``): reads ``central_fluxes``, maps +
     converts each Millard reaction onto its v2ecoli FBA reaction id via the
     committed reaction map, and writes ``pinned_flux_targets``
     ({fba_reaction_id -> flux bound}).

``ecoli-metabolism`` then reads ``pinned_flux_targets`` (its new input port) and
hard-pins those reactions before the LP solve, relaxing any pin that makes the
LP infeasible (recorded in ``listeners.fba_bridge.relaxed_reactions``).

Tick order (enforced by inject_flow_dependencies layer tokens):
    millard-pdmp-metabolism -> fba-flux-coupler -> ecoli-metabolism

Topology overview
-----------------
central_fluxes          <- Millard per-reaction fluxes (mM/s)        [agent root]
pinned_flux_targets     <- coupler-produced FBA pins {fba_id: bound}  [agent root]
bridge_diagnostics      <- coupler per-tick diagnostics               [agent root]
shared/lqr_control      -> Millard control input (no LQR here: stays u=0)
shared/central_metabolites <- Millard species concentrations
_millard_sink_bulk      <- Millard bulk writeback DUMP (never read by WCM)
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    CachedConfigLoader,
    _expand_flushes,
    FLUSH,
    ALL_PARTITIONED,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)

# Reuse baseline's step dispatcher + layer/feature definitions wholesale — the
# harness is "baseline + two edges", so everything except the inserted layer
# slot is identical to baseline.
from v2ecoli.composites import baseline as _baseline


# ---------------------------------------------------------------------------
# Edge names handled by our own builders (skip baseline's dispatcher).
# ---------------------------------------------------------------------------
MILLARD_EDGE = "millard-pdmp-metabolism"
COUPLER_EDGE = "fba-flux-coupler"
_BRIDGE_EDGES = (MILLARD_EDGE, COUPLER_EDGE)


def build_execution_layers(features=None):
    """baseline's layers + the two bridge edges inserted before metabolism."""
    layers = copy.deepcopy(_baseline.BASE_EXECUTION_LAYERS)

    # Insert the Millard ODE + coupler (each followed by a FLUSH) immediately
    # before the ecoli-metabolism layer.
    for i, layer in enumerate(layers):
        if isinstance(layer, list) and "ecoli-metabolism" in layer:
            layers[i:i] = [[MILLARD_EDGE], FLUSH, [COUPLER_EDGE], FLUSH]
            break
    else:  # pragma: no cover - baseline always has ecoli-metabolism
        raise RuntimeError("ecoli-metabolism layer not found in baseline layers")

    # Apply the same feature-module insertion logic baseline uses.
    for feat_name in (features or []):
        feat = _baseline.FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        if "insert_after" in feat:
            ref = feat["insert_after"]
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get("steps", []):
                        layers.insert(i + 1, [step_name])
                    break
        if "insert_before" in feat:
            ref = feat["insert_before"]
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get("steps", [])):
                        layers.insert(i, [step_name])
                    break
        for listener in feat.get("listeners", []):
            for layer in layers:
                if isinstance(layer, list) and "ecoli-mass-listener" in layer:
                    if listener not in layer:
                        layer.append(listener)
                    break
    return _expand_flushes(layers)


# ---------------------------------------------------------------------------
# Bridge edge builders
# ---------------------------------------------------------------------------

def _build_millard_flux_source_edge(core: Any, *, tick_s: float = 1.0):
    """Build the Millard ODE edge used ONLY as a central_fluxes source.

    The bulk writeback port is routed to a throwaway sink store so the Millard
    ODE never modifies the WCM's structured bulk array — the real
    ``ecoli-metabolism`` remains the sole metabolic bulk driver. lqr_control is
    wired to ``shared/lqr_control`` (seeded to u=0): no LQR loop runs here, the
    ODE simply advances open-loop and reports its reaction fluxes.
    """
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    cfg = {
        "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
        "tick_s": tick_s,
        "intervals": 10,
        "control_reaction": "PTS_4",
        "control_parameter": "kF",
        "u_clip": 0.5,
    }
    instance = MillardPDMPMetabolism(config=cfg, core=core)
    in_topo = {
        "lqr_control": ("shared", "lqr_control"),
        # Sink bulk so the writeback (if any) never touches the WCM bulk.
        "bulk": ("_millard_sink_bulk",),
        "listeners_mass": ("listeners", "mass"),
    }
    out_topo = {
        "species_concentrations": ("shared", "central_metabolites"),
        # The only output the bridge actually consumes.
        "central_fluxes": ("central_fluxes",),
        "control_applied": ("shared", "control_applied"),
        "bulk": ("_millard_sink_bulk",),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type="process", config=cfg,
    )
    edge["interval"] = tick_s
    return edge


def _build_fba_flux_coupler_edge(core: Any, *, tick_s: float = 1.0,
                                 coefficient: float = 1.0,
                                 reaction_map_file: str | None = None):
    """Build the FBAFluxCoupler edge: central_fluxes -> pinned_flux_targets."""
    from v2ecoli.steps.fba_flux_coupler import FBAFluxCoupler
    cfg: dict[str, Any] = {"coefficient": float(coefficient)}
    if reaction_map_file is not None:
        cfg["reaction_map_file"] = reaction_map_file
    instance = FBAFluxCoupler(config=cfg, core=core)
    in_topo = {"central_fluxes": ("central_fluxes",)}
    out_topo = {
        "pinned_flux_targets": ("pinned_flux_targets",),
        "bridge_diagnostics": ("bridge_diagnostics",),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type="process", config=cfg,
    )
    edge["interval"] = tick_s
    return edge


def _register_bridge_links(core):
    """core_extensions hook — register the coupler link (the Millard edge is
    stored as a direct instance by make_edge, so it needs no link registry)."""
    from v2ecoli.steps.fba_flux_coupler import register as register_coupler
    register_coupler(core)
    return core


# ---------------------------------------------------------------------------
# Composite generator
# ---------------------------------------------------------------------------

@composite_generator(
    name="millard_fba_bridge_harness",
    description=(
        "Whole-cell E. coli baseline (real tFBA ecoli-metabolism) with the "
        "Millard 2017 central-carbon ODE wired in as a flux SOURCE: the Millard "
        "fluxes are mapped+converted by fba-flux-coupler into pinned_flux_targets "
        "that ecoli-metabolism hard-pins before its LP solve. Integration climax "
        "of the FBA-bridge flux-pin feature."
    ),
    parameters={
        "seed": {
            "type": "integer", "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string", "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "tick_s": {
            "type": "float", "default": 1.0,
            "description": "Millard / coupler update interval in seconds",
        },
        "coefficient": {
            "type": "float", "default": 1.0,
            "description": (
                "Scalar coefficient applied by FBAFluxCoupler when converting "
                "a Millard flux (mM/s) into an FBA reaction-flux bound."
            ),
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    core_extensions=[_register_bridge_links],
)
def millard_fba_bridge_harness(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    tick_s: float = 1.0,
    coefficient: float = 1.0,
) -> dict:
    """Build the process-bigraph document for the Millard FBA-bridge harness."""
    if core is None:
        core = build_core()
    _register_bridge_links(core)

    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    features = list(_baseline.DEFAULT_FEATURES)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    for store in ["listeners", "process", "allocator_rng", "process_state",
                  "exchange", "next_update_time"]:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault("global_time", 0.0)
    cell_state.setdefault("timestep", 1.0)
    cell_state.setdefault("divide", False)
    cell_state.setdefault("division_threshold", "mass_distribution")
    cell_state.setdefault("listeners", {})
    cell_state["listeners"].setdefault("mass", {"dry_mass": 0.0, "cell_mass": 0.0})
    cell_state.setdefault("allocator_rng", np.random.RandomState(seed=seed))

    cell_state.setdefault("ppgpp_state", {"basal_prob": [], "frac_active_rnap": 0.0})
    cell_state.setdefault("attenuation_config", {"enabled": False})

    nut = cell_state.setdefault("next_update_time", {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    cell_state.setdefault("request", {})
    cell_state.setdefault("allocate", {})
    for proc_name in ALL_PARTITIONED:
        cell_state["request"].setdefault(proc_name, {"bulk": {}})
        cell_state["allocate"].setdefault(proc_name, {"bulk": {}})
    n_part = len(ALL_PARTITIONED)
    cell_state["listeners"].setdefault("atp", {
        "atp_requested": np.zeros(n_part, dtype=int),
        "atp_allocated_initial": np.zeros(n_part, dtype=int),
    })

    cell_state.setdefault("process_state", {})
    cell_state["process_state"].setdefault("polypeptide_elongation", {
        "aa_exchange_rates": np.zeros(21),
        "gtp_to_hydrolyze": 0,
        "aa_count_diff": np.zeros(21),
    })

    # FBA-bridge stores (agent root).
    cell_state.setdefault("central_fluxes", {})
    cell_state.setdefault("bridge_diagnostics", {})

    # pinned_flux_targets store: ecoli-metabolism declares this input as
    # `map[float]`, and Map wins the schema resolution against the coupler's
    # InPlaceDict output (see v2ecoli/types/__init__._resolve_map_inplace). A
    # `map[float]` apply only updates keys ALREADY present in the store — a
    # plain-dict update with new keys is silently dropped. So we pre-seed the
    # store with every FBA reaction id in the coupler's reaction-map image
    # (value 0.0) so the per-tick pin writes land on existing keys. (On tick 1
    # the Millard ODE has not yet published fluxes, so these stay 0.0 and the
    # corresponding reactions are pinned to zero then relaxed if infeasible;
    # from tick 2 on the coupler overwrites them with the real ODE-derived
    # bounds.)
    from v2ecoli.library.fba_reaction_map import load_reaction_map
    _rmap = load_reaction_map(
        "v2ecoli/data/millard_v2ecoli_reaction_map.yaml")
    _pin_seed = {
        fba_id: 0.0
        for entries in _rmap.values()
        for (fba_id, _scale) in entries
    }
    cell_state.setdefault("pinned_flux_targets", _pin_seed)

    # Millard shared stores.
    cell_state.setdefault("shared", {})
    cell_state["shared"].setdefault("central_metabolites", {})
    cell_state["shared"].setdefault("lqr_control", {"u": 0.0, "u_dict": {}})
    cell_state["shared"].setdefault("control_applied", {})

    loader = CachedConfigLoader(
        configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    for step_name in flow_order:
        if step_name == MILLARD_EDGE:
            cell_state[step_name] = _build_millard_flux_source_edge(
                core, tick_s=tick_s)
            continue
        if step_name == COUPLER_EDGE:
            cell_state[step_name] = _build_fba_flux_coupler_edge(
                core, tick_s=tick_s, coefficient=coefficient)
            continue
        config = _baseline._get_step_config(
            loader, step_name, core, _process_cache, master_seed=seed)
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(
                    instance, topology, edge_type=edge_type)

    for proc_name, proc_instance in _process_cache.items():
        cell_state["process"][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    seed_mass_listener(cell_state, core)

    inject_flow_dependencies(cell_state, flow_order, layers=execution_layers)

    state = {"agents": {"0": cell_state}, "global_time": 0.0}

    return {
        "state": state,
        "skip_initial_steps": True,
        "sequential_steps": False,
        "flow_order": flow_order,
    }
