"""Baseline-composite recipes for the dnaa-replication investigation.

Each recipe is a thin wrapper that takes the upstream ``baseline`` composite
and adds the specific Steps for one study's gate. The recipes are registered
via :func:`pbg_superpowers.composite_generator.composite_generator` so the
dashboard discovers them.

Recipes shipped here:

* ``dnaa_02_with_intrinsic_hydrolysis`` (this file) — adds the three Steps
  required by study ``dnaa-02-atp-hydrolysis``:

    - ``DnaaIntrinsicHydrolysis`` (DnaA-ATP -> DnaA-ADP, k = 0.046/min)
    - ``DnaaAtpFractionClamp`` (ATP-fraction in [low, high]; stand-in for
      the deferred RIDA / DDAH / DARS network)
    - ``DnaaCycleListener`` (apo/ATP/ADP counts + fractions per tick)

The wrapper preserves baseline.py's execution-layer wiring; the new Steps
are appended at the END of the flow with explicit low priorities so they
run AFTER ecoli-equilibrium has rebound the DnaA-ATP equilibrium. The
``DnaaCycleListener`` runs LAST so its emit reflects the post-hydrolysis,
post-clamp state.

See ``studies/dnaa-02-atp-hydrolysis/study.yaml`` for the test plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import make_edge
from v2ecoli.composites.baseline import baseline
from v2ecoli.steps.dnaa_intrinsic_hydrolysis import DnaaIntrinsicHydrolysis
from v2ecoli.steps.dnaa_cycle_listener import DnaaCycleListener
from v2ecoli.steps.dnaa_atp_fraction_clamp import DnaaAtpFractionClamp
from v2ecoli.processes.dnaa_box_binding import DnaaBoxBinding


# Priority slots for the appended steps (lower = runs later in the tick,
# by process-bigraph convention). All are below the smallest baseline
# priority (~1.0 for the last step in the layer chain), so they fire
# after every baseline step has settled.
_PRIORITY_HYDROLYSIS  = 0.30
_PRIORITY_CLAMP       = 0.20
_PRIORITY_BOX_BINDING = 0.15  # between clamp and listener
_PRIORITY_LISTENER    = 0.10


_DNAA_ADP_REACTION_ID = "MONOMER0-4565_RXN"


def _disable_dnaa_adp_equilibrium(cell_state):
    """Zero out the MONOMER0-4565_RXN reaction in the ecoli-equilibrium Step.

    Mutates the live Step instance's ``stoichMatrix`` so the reaction's
    forward and reverse fluxes have no effect on bulk counts. The ODE
    solver may still compute a flux for the reaction, but since the
    stoichiometry column is zero the result is a zero delta for every
    participating molecule. See ``dnaa-02-EQ-04`` in the study yaml
    for the design discussion.

    Raises a clean error if the ecoli-equilibrium Step or the named
    reaction can't be located, so a misconfigured recipe fails loudly
    rather than silently leaving the tug-of-war in place.
    """
    edge = cell_state.get("ecoli-equilibrium")
    if not isinstance(edge, dict) or "instance" not in edge:
        raise RuntimeError(
            "_disable_dnaa_adp_equilibrium: ecoli-equilibrium step is "
            "not in cell_state. Did baseline() change its step name?"
        )
    eq = edge["instance"]
    rxn_ids = list(getattr(eq, "reaction_ids", []))
    if _DNAA_ADP_REACTION_ID not in rxn_ids:
        raise RuntimeError(
            f"_disable_dnaa_adp_equilibrium: {_DNAA_ADP_REACTION_ID!r} not "
            f"in ecoli-equilibrium.reaction_ids. Was sim_data updated?"
        )
    idx = rxn_ids.index(_DNAA_ADP_REACTION_ID)
    # Zero the reaction's column in S so S @ nu produces no molecule delta.
    eq.stoichMatrix = eq.stoichMatrix.copy()
    eq.stoichMatrix[:, idx] = 0
    # Zero the forward/reverse rate constants if the inline ODE path is
    # in use (the cached config in this workspace uses the legacy opaque
    # ``fluxesAndMoleculesToSS`` callable; rates_fwd / rates_rev are then
    # empty arrays, which we leave alone).
    if hasattr(eq, "rates_fwd") and len(getattr(eq, "rates_fwd", [])) > idx:
        eq.rates_fwd = np.array(eq.rates_fwd, copy=True)
        eq.rates_rev = np.array(eq.rates_rev, copy=True)
        eq.rates_fwd[idx] = 0.0
        eq.rates_rev[idx] = 0.0


def _append_step(cell_state, step_name, instance, priority,
                 token_in=None, token_out=None):
    """Add a step instance to cell_state with explicit low priority + tokens.

    process-bigraph's execution order is dependency-driven: a Step that
    reads a store written by another Step runs after that other Step.
    Priority is a tie-breaker for *independent* Steps, not a guaranteed
    ordering between Steps that touch overlapping stores.

    To force hydrolysis -> clamp -> listener ordering within a tick we
    add explicit per-Step token stores: hydrolysis writes
    ``_dnaa02_token_h``, the clamp reads it and writes
    ``_dnaa02_token_c``, the listener reads ``_dnaa02_token_c``. The
    tokens are dummy floats; the framework treats them as a write→read
    dependency and serializes the three Steps in that order.
    """
    edge = make_edge(instance, instance.topology, edge_type='step')
    edge['priority'] = priority
    if token_in is not None:
        edge.setdefault('inputs', {})[f'_dep_in_{step_name}'] = [token_in]
    if token_out is not None:
        edge.setdefault('outputs', {})[f'_dep_out_{step_name}'] = [token_out]
    cell_state[step_name] = edge


@composite_generator(
    name="dnaa_02_with_intrinsic_hydrolysis",
    description=(
        "v2ecoli baseline + DnaA nucleotide cycle (intrinsic hydrolysis, "
        "atp-fraction clamp, dnaA_cycle listener). dnaa-02 gate."
    ),
    parameters={
        "seed":          {"type": "integer", "default": 0,        "description": "RNG seed"},
        "cache_dir":     {"type": "string",  "default": "out/cache",
                          "description": "Path to ParCa cache directory"},
        "hydrolysis_rate_per_min": {
            "type": "number", "default": 0.046,
            "description": "First-order DnaA-ATP -> DnaA-ADP rate (Sekimizu 1987).",
        },
        "hydrolysis_deterministic": {
            "type": "boolean", "default": False,
            "description": "If true, the hydrolysis Step rounds the expected "
                           "transfer instead of drawing Poisson. Used in tests.",
        },
        "atp_fraction_clamp_low": {
            "type": "number", "default": None,
            "description": "Lower edge of the ATP-fraction clamp band. "
                           "Pair with atp_fraction_clamp_high to enable.",
        },
        "atp_fraction_clamp_high": {
            "type": "number", "default": None,
            "description": "Upper edge of the ATP-fraction clamp band.",
        },
    },
)
def dnaa_02_with_intrinsic_hydrolysis(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    hydrolysis_rate_per_min: float = 0.046,
    hydrolysis_deterministic: bool = False,
    atp_fraction_clamp_low: float | None = None,
    atp_fraction_clamp_high: float | None = None,
) -> dict:
    """Build the dnaa-02 gate composite.

    Args:
        core: bigraph-schema core. If None, baseline() creates one.
        seed: RNG seed for stochastic init.
        cache_dir: ParCa cache directory.
        hydrolysis_rate_per_min: DnaA-ATP -> DnaA-ADP rate. Default
            0.046/min from Sekimizu 1987.
        hydrolysis_deterministic: round expected transfers (tests).
        atp_fraction_clamp_low, atp_fraction_clamp_high: if both set,
            the clamp Step holds DnaA-ATP / total inside [low, high].
            Both ``None`` disables the clamp.

    Returns:
        process-bigraph state document.
    """
    # 1. Build the standard baseline composite.
    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    # 1b. BIOLOGY PIVOT (resolved dnaa-02-EQ-04, option B):
    #     Disable the existing MONOMER0-4565_RXN equilibrium reaction
    #
    #         PD03831[c] + ADP[c]  ->  MONOMER0-4565[c]
    #
    #     so that DnaA-ADP can only form via the hydrolysis pathway
    #     (intrinsic in this study; RIDA/DDAH/DARS in dnaa-05). Without
    #     this patch, ecoli-equilibrium drives DnaA-ATP/DnaA-ADP back to
    #     its thermodynamic ratio each tick and reverses our hydrolysis
    #     and clamp transfers.
    #
    #     Biological justification: in vivo, the DnaA-ADP pool comes
    #     overwhelmingly from hydrolysis of bound DnaA-ATP, not from
    #     direct binding of free DnaA to free ADP. Sekimizu 1987 +
    #     Kawakami 2006 treat the productive cycle as ATP-binding ->
    #     intrinsic / RIDA / DDAH hydrolysis. The ADP-binding
    #     equilibrium is a minor biochemical side reaction that
    #     happens to be encoded in v2ecoli's sim_data.
    #
    #     MONOMER0-160_RXN (the ATP-binding equilibrium) is INTACT.
    _disable_dnaa_adp_equilibrium(cell_state)

    # 2. Instantiate the three dnaa-02 Steps.
    hydrolysis = DnaaIntrinsicHydrolysis(parameters={
        "rate_per_min":   float(hydrolysis_rate_per_min),
        "deterministic":  bool(hydrolysis_deterministic),
        "seed":           int(seed),
    })

    clamp_band = None
    if (atp_fraction_clamp_low is not None
            and atp_fraction_clamp_high is not None):
        clamp_band = [float(atp_fraction_clamp_low),
                      float(atp_fraction_clamp_high)]
    clamp = DnaaAtpFractionClamp(parameters={"band": clamp_band})

    listener = DnaaCycleListener(parameters={})

    # 3. Append edges with explicit low priorities AND token chains so the
    #    three new Steps fire in order hydrolysis -> clamp -> listener.
    #    Token stores are seeded as plain floats in cell_state so the
    #    framework can hash them.
    cell_state.setdefault("_dnaa02_token_h", 0.0)
    cell_state.setdefault("_dnaa02_token_c", 0.0)

    _append_step(cell_state, "dnaa-intrinsic-hydrolysis",
                 hydrolysis, _PRIORITY_HYDROLYSIS,
                 token_out="_dnaa02_token_h")
    _append_step(cell_state, "dnaa-atp-fraction-clamp",
                 clamp, _PRIORITY_CLAMP,
                 token_in="_dnaa02_token_h",
                 token_out="_dnaa02_token_c")
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER,
                 token_in="_dnaa02_token_c")

    # 4. Extend flow_order so listing tools (network_report etc.) include them.
    doc["flow_order"] = list(doc.get("flow_order", [])) + [
        "dnaa-intrinsic-hydrolysis",
        "dnaa-atp-fraction-clamp",
        "dnaa-cycle-listener",
    ]

    return doc


# ─── dnaa-03: box binding ──────────────────────────────────────────────────
@composite_generator(
    name="dnaa_03_with_box_binding",
    description=(
        "v2ecoli baseline + dnaa-02 nucleotide cycle + dnaa-03 DnaA-box "
        "binding (322+ catalogued sites: 11 oriC + 7 dnaAp + 307 chromosomal "
        "+ datA/DARS placeholders)."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0},
        "cache_dir": {"type": "string",  "default": "out/cache"},
        # dnaa-02 params (inherited)
        "hydrolysis_rate_per_min": {"type": "number", "default": 0.046},
        "atp_fraction_clamp_low":  {"type": "number", "default": 0.2},
        "atp_fraction_clamp_high": {"type": "number", "default": 0.5},
        # dnaa-03 params
        "kd_high_nM":           {"type": "number",  "default": 1.0},
        "kd_low_nM":            {"type": "number",  "default": 100.0},
        "enable_oric_binding":  {"type": "boolean", "default": True},
        "enable_dnaap_binding": {"type": "boolean", "default": True},
        "initial_dnaA_count_per_cell": {
            "type": "integer", "default": None,
            "description": "Override initial DnaA count per cell. Use ~500 "
                           "to test dnaa-03's titration biology at the "
                           "literature DnaA level instead of the 5x-low "
                           "dnaa-01 calibration shortfall (~115).",
        },
    },
)
def dnaa_03_with_box_binding(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    hydrolysis_rate_per_min: float = 0.046,
    atp_fraction_clamp_low: float | None = 0.2,
    atp_fraction_clamp_high: float | None = 0.5,
    kd_high_nM: float = 1.0,
    kd_low_nM: float = 100.0,
    enable_oric_binding: bool = True,
    enable_dnaap_binding: bool = True,
    initial_dnaA_count_per_cell: int | None = None,
) -> dict:
    """Build the dnaa-03 gate composite.

    Stacks on top of dnaa-02 (intrinsic hydrolysis + ATP-fraction clamp,
    with MONOMER0-4565_RXN disabled) and adds the box-binding Process
    that tracks per-site occupancy across 322 active DnaA-boxes.

    The dnaa-02 default clamp band [0.2, 0.5] is on by default here so
    the box-binding model sees a physiologically reasonable ATP/ADP
    free-pool split. Override with ``atp_fraction_clamp_low=None``
    if you want the intrinsic-only behavior.
    """
    doc = dnaa_02_with_intrinsic_hydrolysis(
        core=core, seed=seed, cache_dir=cache_dir,
        hydrolysis_rate_per_min=hydrolysis_rate_per_min,
        atp_fraction_clamp_low=atp_fraction_clamp_low,
        atp_fraction_clamp_high=atp_fraction_clamp_high,
    )
    cell_state = doc["state"]["agents"]["0"]

    box_binding = DnaaBoxBinding(parameters={
        "kd_high_nM":           float(kd_high_nM),
        "kd_low_nM":            float(kd_low_nM),
        "seed":                 int(seed),
        "enable_oric_binding":  bool(enable_oric_binding),
        "enable_dnaap_binding": bool(enable_dnaap_binding),
    })

    _append_step(cell_state, "dnaa-box-binding",
                 box_binding, _PRIORITY_BOX_BINDING)

    doc["flow_order"] = list(doc.get("flow_order", [])) + ["dnaa-box-binding"]

    # Optional override: boost initial DnaA pool. The dnaa-01 baseline
    # produces ~115/cell which is 5× below the literature [300, 800] band.
    # When testing dnaa-03's titration biology, the chromosome's 307 sites
    # can never reach 0.9 occupancy with only 115 DnaA — the system is
    # DnaA-limited, not Kd-limited. This override bypasses the dnaa-01
    # calibration shortfall so we can test dnaa-03's mechanism in isolation.
    # The initial DnaA is added to bulk[MONOMER0-160[c]] (DnaA-ATP form)
    # at composite build time; the dnaa-02 cycle (hydrolysis + clamp) will
    # redistribute it across ATP/ADP states within the first ~10 minutes.
    if initial_dnaA_count_per_cell is not None:
        _set_initial_dnaA_pool(cell_state, int(initial_dnaA_count_per_cell))

    return doc


def _set_initial_dnaA_pool(cell_state, target_total: int) -> None:
    """Override the initial DnaA bulk pool to a target total count.

    Distributes the target across MONOMER0-160 (ATP-bound) by default,
    leaving PD03831 (apo) and MONOMER0-4565 (ADP-bound) at their current
    values. The dnaa-02 cycle Steps redistribute the pool to the clamp
    band [0.2, 0.5] within the first minutes of the run.
    """
    import numpy as np
    bulk = cell_state["bulk"]
    ids = bulk["id"]
    atp_idx = np.where(ids == "MONOMER0-160[c]")[0]
    apo_idx = np.where(ids == "PD03831[c]")[0]
    adp_idx = np.where(ids == "MONOMER0-4565[c]")[0]
    if not (len(atp_idx) and len(apo_idx) and len(adp_idx)):
        raise RuntimeError(
            "_set_initial_dnaA_pool: one of PD03831[c] / MONOMER0-160[c] / "
            "MONOMER0-4565[c] not found in bulk."
        )
    current = int(bulk[atp_idx[0]]["count"]) + int(bulk[apo_idx[0]]["count"]) \
              + int(bulk[adp_idx[0]]["count"])
    delta = target_total - current
    # Put the delta in the ATP-complex form — the dnaa-02 cycle will
    # split it physically.
    bulk["count"][atp_idx[0]] = int(bulk[atp_idx[0]]["count"]) + delta
