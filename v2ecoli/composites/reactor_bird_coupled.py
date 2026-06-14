"""reactor_bird_coupled — baseline_population + BiRD reactor + cell<->reactor coupler.

Build-phase wire-up for mbp-03-reactor-coupling (req-2). This is the
multiscale-bioprocess composite: it composes v2ecoli's single-cell whole-cell
model (with the population aggregator) against the BiRD bioreactor transport
physics, mediated by the :class:`ReactorCellCoupler` Step.

Topology (Option B of the v2ecoli <-> pbg-bioreactordesign coupling — BiRD owns
transport, v2ecoli owns biomass + metabolic demand)::

    baseline_population (cell side)
      agents.0.*            single-cell WCM
      population.*          PopulationAggregator output (biomass_concentration_gL, ...)
      environment.*         top-level env store (external_concentrations, media_id)
        environment_driver  EnvironmentDriver in external_store mode (no-op; the
                            coupler is the env source of truth)
        environment_mirror  propagates environment.external_concentrations ->
                            agents.*.boundary.external each tick

    reactor.*               shared reactor stores (mg/L gases, g/L biomass)
      reactor_transport     BiRDTransportProcess (local:BiRDTransportProcess)
      reactor_cell_coupler  ReactorCellCoupler Step

Shared-store / additive-delta convention (mirrors
``pbg_bioreactordesign.composites._factory.make_coupled_document``):

  * ``reactor.dissolved_o2`` / ``reactor.dissolved_co2`` are bare ``float``
    leaves. BOTH BiRDTransportProcess (transport, +) and ReactorCellCoupler
    (consumption, -) write ADDITIVE deltas to them; process-bigraph's ``float``
    ``_apply`` sums the two contributions, so the net change each tick is
    ``transport - consumption``.
  * ``reactor.biomass`` is ``overwrite[float]``: the coupler passes the
    population biomass concentration through (absolute value), and
    BiRDTransportProcess reads it as a read-only input.

Port-schema subtlety (load-bearing — verified empirically against the PB
realize/project path): ReactorCellCoupler declares a single ``reactor`` output
port typed ``InPlaceDict`` (an empty Node). When the ``reactor`` store becomes a
structured tree (because BiRDTransportProcess wires ``dissolved_o2`` /
``dissolved_co2`` as concrete ``float`` leaf outputs), PB projects the coupler's
update per-leaf and DROPS any leaf the port schema doesn't enumerate — so
``reactor.biomass`` would silently never apply. We therefore OVERRIDE the
coupler edge's ``_outputs['reactor']`` with an explicit per-leaf schema
(``biomass: overwrite[float]``, ``dissolved_o2/co2: float``) so biomass routes
(overwrite) while the gases stay additive. The gases alone propagate even
without the override (they have transport leaf-output ports), but biomass does
not; the override fixes both consistently.
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.composites.baseline_population import baseline_population
from v2ecoli.composites.baseline_time_varying_env import (
    ENVIRONMENT_DRIVER_STEP_NAME,
    ENVIRONMENT_MIRROR_STEP_NAME,
    _empty_environment_store,
)
from v2ecoli.steps.environment_driver import (
    ENV_DRIVER_MODE_EXTERNAL_STORE,
    EnvironmentDriver,
)
from v2ecoli.steps.environment_mirror import EnvironmentMirror
from v2ecoli.steps.reactor_cell_coupler import ReactorCellCoupler


# Top-level state keys / step names.
REACTOR_STORE_NAME = "reactor"
REACTOR_TRANSPORT_NAME = "reactor_transport"
REACTOR_CELL_COUPLER_STEP_NAME = "reactor_cell_coupler"


# Default BiRDTransportProcess config (mbp-03 req-2). Time base: hours.
DEFAULT_BIRD_REACTOR_CONFIG: dict[str, Any] = {
    "reactor_type":      "bubble_column",
    "volume_L":          1.0,
    "gas_flow_rate_Lpm": 2.0,
    "temperature_K":     310.15,
}

# Fallback initial dissolved-gas concentrations (mg/L), used only if the BiRD
# equilibrium can't be computed at build time. ~O2 saturation at 37 C.
FALLBACK_DISSOLVED_O2_MGL = 8.0
FALLBACK_DISSOLVED_CO2_MGL = 0.5

# BiRDTransportProcess update interval (HOURS — the reactor physics time base).
#
# IMPORTANT (stability): BiRDTransportProcess.update applies an explicit-Euler
# transport delta ``dC = kLa*(C* - C)*interval``. The discrete map is stable
# only when ``kLa*interval < ~1`` (otherwise a deviation is amplified ~kLa*dt
# per step and the dissolved-gas store diverges/oscillates). For the default
# bubble_column config kLa_O2 ~= 20.8 /h, so an interval of 1.0 h is wildly
# unstable; 0.01 h gives a damping factor ~0.2 and the reactor relaxes to the
# gas-liquid equilibrium monotonically.
#
# NOTE (time base): the WCM steps in SECONDS while BiRD is in HOURS. process-
# bigraph feeds this interval (in global-time units = seconds) straight into the
# transport math as hours, so the reactor over-transports relative to wall-clock
# (it holds the dissolved gases near saturation each cell tick). The proper
# seconds->hours bridge (a thin wrapper, mirroring the coupler's own /3600
# conversion) is mbp-03 T5 follow-up; this composite's contract is build + a
# finite single coupled step + correct ADDITIVE store sharing, all of which hold.
DEFAULT_TRANSPORT_INTERVAL = 0.01


def _transport_equilibrium(bird_config: dict[str, Any]) -> tuple[float, float]:
    """Gas-liquid equilibrium dissolved O2/CO2 (mg/L) for the BiRD config.

    Seeding the shared stores at the transport fixed point (``C* ``) means the
    reactor starts at gas-liquid equilibrium, so the only initial dynamics come
    from the cell's metabolic demand rather than an artificial saturation
    transient. Falls back to fixed values if the BiRD transport module isn't
    importable (the composite can't actually run in that case, but the document
    still builds).
    """
    try:
        from pbg_bioreactordesign import BiRDTransportProcess
        from pbg_bioreactordesign.transport import compute_transport_state

        cfg = {k: spec.get("_default")
               for k, spec in BiRDTransportProcess.config_schema.items()}
        cfg.update(bird_config)
        t = compute_transport_state(cfg, cfg["gas_flow_rate_Lpm"])
        return float(t["cstar_o2"]), float(t["cstar_co2"])
    except Exception:
        return FALLBACK_DISSOLVED_O2_MGL, FALLBACK_DISSOLVED_CO2_MGL


def _reactor_store(bird_config: dict[str, Any]) -> dict[str, Any]:
    """Seed the shared reactor store.

    ``dissolved_o2`` / ``dissolved_co2`` are bare floats (additive multi-writer
    shared stores). ``biomass`` is overwrite[float] (single writer = coupler,
    passthrough). Read-only inputs (``glucose``, ``gas_flow_rate_Lpm``,
    ``volume_L``) and the transport diagnostics are seeded so wires resolve.
    """
    cstar_o2, cstar_co2 = _transport_equilibrium(bird_config)
    return {
        # Additive shared dissolved-gas stores (mg/L) — both transport and the
        # coupler write deltas; the float _apply sums them. Seeded at the
        # gas-liquid equilibrium (C*) so the transport delta starts near zero.
        "dissolved_o2":      cstar_o2,
        "dissolved_co2":     cstar_co2,
        # Overwrite passthrough: coupler writes, transport reads (g/L).
        "biomass":           {"_type": "overwrite[float]", "_default": 0.0},
        # Read-only transport inputs.
        "glucose":           0.0,
        "gas_flow_rate_Lpm": float(bird_config.get("gas_flow_rate_Lpm", 2.0)),
        # BiRD owns volume; the coupler reads it (BiRD config seeds it).
        "volume_L":          float(bird_config.get("volume_L", 1.0)),
        # Transport diagnostics (overwrite readouts).
        "o2_transport_delta":  {"_type": "overwrite[float]", "_default": 0.0},
        "co2_transport_delta": {"_type": "overwrite[float]", "_default": 0.0},
        "kla_o2":              {"_type": "overwrite[float]", "_default": 0.0},
        "o2_saturation":       {"_type": "overwrite[float]", "_default": 0.0},
        "co2_saturation":      {"_type": "overwrite[float]", "_default": 0.0},
        "gas_holdup":          {"_type": "overwrite[float]", "_default": 0.0},
    }


@composite_generator(
    name="reactor_bird_coupled",
    description=(
        "v2ecoli baseline_population coupled to the BiRD bioreactor "
        "(BiRDTransportProcess) via ReactorCellCoupler. The cell population's "
        "metabolic O2/CO2 exchange and the reactor's gas-liquid transport net "
        "additively at shared reactor.dissolved_o2/dissolved_co2 stores; the "
        "EnvironmentDriver runs in external_store mode (the coupler is the env "
        "source) and EnvironmentMirror propagates the reactor-derived boundary "
        "concentrations to every agent."
    ),
    parameters={
        "seed":            {"type": "int",    "default": 0},
        "cache_dir":       {"type": "string", "default": "out/cache"},
        # BiRDTransportProcess config (reactor_type / volume_L /
        # gas_flow_rate_Lpm / temperature_K). Time base hours.
        "bird_reactor_config": {
            "type": "object", "default": dict(DEFAULT_BIRD_REACTOR_CONFIG)},
        # Population-aggregator knobs forwarded to baseline_population.
        "cells_per_agent": {"type": "number", "default": 1.0},
    },
)
def reactor_bird_coupled(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    bird_reactor_config: dict | None = None,
    cells_per_agent: float = 1.0,
) -> dict:
    """Build the reactor_bird_coupled document.

    Extends ``baseline_population`` (cell side) with the BiRD reactor transport
    process + the reactor<->cell coupler, and switches the environment driver to
    ``external_store`` mode with the mirror active.
    """
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    bird_config = dict(DEFAULT_BIRD_REACTOR_CONFIG)
    if bird_reactor_config:
        bird_config.update(bird_reactor_config)

    # --- cell side: baseline + PopulationAggregator -----------------------
    document = baseline_population(
        core, seed=seed, cache_dir=cache_dir, cells_per_agent=cells_per_agent,
        reactor_volume_L=float(bird_config.get("volume_L", 1.0)),
    )
    state = document["state"]
    flow_order = document.setdefault("flow_order", [])

    # --- environment hook: driver (external_store) + mirror ---------------
    # The coupler is the env source of truth, so the driver is a no-op
    # (external_store mode); the mirror must still run so agents see the
    # coupler-written boundary concentrations.
    state.setdefault("environment", _empty_environment_store())

    driver_config = {
        "env_driver_mode":           ENV_DRIVER_MODE_EXTERNAL_STORE,
        "synthetic_trajectory_spec": {},
    }
    driver = _make_instance(EnvironmentDriver, driver_config, core)
    state[ENVIRONMENT_DRIVER_STEP_NAME] = make_edge(
        driver, EnvironmentDriver.topology, edge_type="step",
        config=driver_config,
    )

    mirror = _make_instance(EnvironmentMirror, {}, core)
    state[ENVIRONMENT_MIRROR_STEP_NAME] = make_edge(
        mirror, EnvironmentMirror.topology, edge_type="step", config={},
    )

    # Insert driver + mirror BEFORE the unique_update FLUSH that precedes
    # media_update, so the FLUSH commits their writes before exchange_data
    # re-derives metabolism's exchange constraints (identical placement logic
    # to baseline_time_varying_env).
    if "media_update" in flow_order:
        media_idx = flow_order.index("media_update")
        flush_idx = media_idx - 1
        while flush_idx > 0 and not flow_order[flush_idx].startswith("unique_update_"):
            flush_idx -= 1
        insert_at = (flush_idx
                     if flow_order[flush_idx].startswith("unique_update_")
                     else media_idx)
        flow_order.insert(insert_at, ENVIRONMENT_MIRROR_STEP_NAME)
        flow_order.insert(insert_at, ENVIRONMENT_DRIVER_STEP_NAME)
    else:
        flow_order.extend([ENVIRONMENT_DRIVER_STEP_NAME, ENVIRONMENT_MIRROR_STEP_NAME])

    # --- reactor side: shared stores + transport + coupler ----------------
    state[REACTOR_STORE_NAME] = _reactor_store(bird_config)

    # BiRDTransportProcess (local: link registered in build_core). Reads the
    # shared dissolved stores + biomass/glucose/gas_flow; emits ADDITIVE deltas
    # to dissolved_o2/co2 (bare float ports) + overwrite diagnostics.
    state[REACTOR_TRANSPORT_NAME] = {
        "_type":   "process",
        "address": "local:BiRDTransportProcess",
        "config":  dict(bird_config),
        "interval": DEFAULT_TRANSPORT_INTERVAL,
        "inputs": {
            "dissolved_o2":      [REACTOR_STORE_NAME, "dissolved_o2"],
            "dissolved_co2":     [REACTOR_STORE_NAME, "dissolved_co2"],
            "biomass":           [REACTOR_STORE_NAME, "biomass"],
            "glucose":           [REACTOR_STORE_NAME, "glucose"],
            "gas_flow_rate_Lpm": [REACTOR_STORE_NAME, "gas_flow_rate_Lpm"],
        },
        "outputs": {
            # Additive transport deltas -> shared stores.
            "dissolved_o2":  [REACTOR_STORE_NAME, "dissolved_o2"],
            "dissolved_co2": [REACTOR_STORE_NAME, "dissolved_co2"],
            # Diagnostics (overwrite readouts).
            "o2_transport_delta":  [REACTOR_STORE_NAME, "o2_transport_delta"],
            "co2_transport_delta": [REACTOR_STORE_NAME, "co2_transport_delta"],
            "kla_o2":              [REACTOR_STORE_NAME, "kla_o2"],
            "o2_saturation":       [REACTOR_STORE_NAME, "o2_saturation"],
            "co2_saturation":      [REACTOR_STORE_NAME, "co2_saturation"],
            "gas_holdup":          [REACTOR_STORE_NAME, "gas_holdup"],
        },
    }

    # ReactorCellCoupler Step. Reads population.biomass_concentration_gL +
    # reactor dissolved gases + agent fluxes; writes reactor deltas (additive
    # gases, overwrite biomass) + environment.external_concentrations.
    coupler_config = {
        "cells_per_agent":  float(cells_per_agent),
        "reactor_volume_L": float(bird_config.get("volume_L", 1.0)),
    }
    coupler = _make_instance(ReactorCellCoupler, coupler_config, core)
    coupler_edge = make_edge(
        coupler, ReactorCellCoupler.topology, edge_type="step",
        config=coupler_config,
    )
    # Override the coupler's reactor output schema so biomass routes (overwrite)
    # while dissolved gases stay additive (see module docstring — the bare
    # InPlaceDict port silently drops biomass once reactor is a structured tree).
    coupler_edge["_outputs"]["reactor"] = {
        "biomass":       "overwrite[float]",
        "dissolved_o2":  "float",
        "dissolved_co2": "float",
    }
    state[REACTOR_CELL_COUPLER_STEP_NAME] = coupler_edge

    # Run the coupler at the END of the flow (after the PopulationAggregator
    # has produced population.biomass_concentration_gL and metabolism has
    # emitted the per-agent exchange fluxes this tick).
    flow_order.append(REACTOR_CELL_COUPLER_STEP_NAME)

    return document
