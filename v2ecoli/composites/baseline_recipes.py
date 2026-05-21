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
from v2ecoli.steps.dnaa_rida_v0 import DnaaRidaV0
from v2ecoli.steps.dnaa_initiation_mechanism import DnaaInitiationMechanism
from v2ecoli.processes.dnaa_box_binding import DnaaBoxBinding


# Priority slots for the appended steps (lower = runs later in the tick,
# by process-bigraph convention). All are below the smallest baseline
# priority (~1.0 for the last step in the layer chain), so they fire
# after every baseline step has settled.
_PRIORITY_HYDROLYSIS  = 0.30
_PRIORITY_RIDA        = 0.28   # between intrinsic and clamp; also writes to MONOMER0-160/4565
_PRIORITY_CLAMP       = 0.20
_PRIORITY_BOX_BINDING = 0.15  # between clamp and listener
_PRIORITY_LISTENER    = 0.10
_PRIORITY_INITIATION  = 0.05   # last: reads other listeners' emits


_DNAA_ADP_REACTION_ID = "MONOMER0-4565_RXN"


def _scale_dnaa_adp_equilibrium(cell_state, scale: float):
    """Multiply the MONOMER0-4565_RXN stoichMatrix column by ``scale``.

    Used by variant B' of dnaa-02f. ``scale=0.0`` is equivalent to
    ``_disable_dnaa_adp_equilibrium`` (full disable, option B); ``scale=1.0``
    leaves the reaction untouched (option E baseline). Values in between
    partially attenuate the equilibrium's recovery flux — letting us
    quantitatively answer "is the reverse rate over-estimated, and if so,
    by how much?"

    The interpretation: with the legacy ``fluxesAndMoleculesToSS`` solver
    (no explicit rates_fwd / rates_rev arrays), scaling the stoichMatrix
    column scales the per-tick delta applied via this reaction by the
    same factor, which is mechanically equivalent to scaling both the
    forward and reverse rate constants by ``scale``.

    Raises a clean error if the ecoli-equilibrium Step or the named
    reaction can't be located, mirroring _disable_dnaa_adp_equilibrium.
    """
    if scale < 0:
        raise ValueError(f"_scale_dnaa_adp_equilibrium: scale must be ≥ 0, got {scale}")
    edge = cell_state.get("ecoli-equilibrium")
    if not isinstance(edge, dict) or "instance" not in edge:
        raise RuntimeError(
            "_scale_dnaa_adp_equilibrium: ecoli-equilibrium step is "
            "not in cell_state. Did baseline() change its step name?"
        )
    eq = edge["instance"]
    rxn_ids = list(getattr(eq, "reaction_ids", []))
    if _DNAA_ADP_REACTION_ID not in rxn_ids:
        raise RuntimeError(
            f"_scale_dnaa_adp_equilibrium: {_DNAA_ADP_REACTION_ID!r} not "
            f"in ecoli-equilibrium.reaction_ids. Was sim_data updated?"
        )
    idx = rxn_ids.index(_DNAA_ADP_REACTION_ID)
    eq.stoichMatrix = eq.stoichMatrix.copy()
    eq.stoichMatrix[:, idx] = eq.stoichMatrix[:, idx] * float(scale)
    if hasattr(eq, "rates_fwd") and len(getattr(eq, "rates_fwd", [])) > idx:
        eq.rates_fwd = np.array(eq.rates_fwd, copy=True)
        eq.rates_rev = np.array(eq.rates_rev, copy=True)
        eq.rates_fwd[idx] = eq.rates_fwd[idx] * float(scale)
        eq.rates_rev[idx] = eq.rates_rev[idx] * float(scale)


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
        "atp-fraction clamp, dnaA_cycle listener). dnaa-02 gate. "
        "Default mechanism is 'rida-v0' (winner of dnaa-02f, equilibrium "
        "intact, RIDA-v0 + clamp). Pass mechanism='monkey-patch' for the "
        "legacy 2026-05-17 option-B behavior."
    ),
    parameters={
        "seed":          {"type": "integer", "default": 0,        "description": "RNG seed"},
        "cache_dir":     {"type": "string",  "default": "out/cache",
                          "description": "Path to ParCa cache directory"},
        "mechanism": {
            "type": "string", "default": "rida-v0",
            "description": "DnaA-ADP-generation mechanism. 'rida-v0' (default, "
                           "canonical per dnaa-02f) keeps the equilibrium "
                           "INTACT and adds a fork-active RIDA-v0 Step that "
                           "supplies the missing extrinsic flux. "
                           "'monkey-patch' is the legacy option-B behavior — "
                           "_disable_dnaa_adp_equilibrium mutates the live "
                           "ecoli-equilibrium step's stoichMatrix.",
        },
        "rida_rate_per_min": {
            "type": "number", "default": 4.6,
            "description": "RIDA-v0 effective rate when ≥1 replisome active "
                           "(only relevant when mechanism='rida-v0'). Default "
                           "4.6/min = literature 100× Sekimizu intrinsic.",
        },
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
        "stage1_dnaa_expression": {
            "type": "boolean", "default": False,
            "description": "Add constitutive DnaA expression + seed apo-DnaA so "
                           "DnaA is at the physiological ~325 (Stage-1). Use with "
                           "cache_dir=out/cache-stage1-heuristic.",
        },
        "dnaa_synthesis_rate_per_min": {
            "type": "number", "default": 1.5,
            "description": "Constitutive DnaA synthesis rate (Stage-1: 1.5/min/gene).",
        },
        "initial_dnaa_count": {
            "type": "integer", "default": 325,
            "description": "Seed apo-DnaA count when stage1_dnaa_expression=True.",
        },
    },
)
def dnaa_02_with_intrinsic_hydrolysis(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    mechanism: str = "rida-v0",
    rida_rate_per_min: float = 4.6,
    hydrolysis_rate_per_min: float = 0.046,
    hydrolysis_deterministic: bool = False,
    atp_fraction_clamp_low: float | None = None,
    atp_fraction_clamp_high: float | None = None,
    stage1_dnaa_expression: bool = False,
    dnaa_synthesis_rate_per_min: float = 1.5,
    initial_dnaa_count: int = 325,
) -> dict:
    """Build the dnaa-02 gate composite.

    ``stage1_dnaa_expression=True`` (expert round-1, 2026-05-21): also add a
    constitutive DnaaConstitutiveExpression Step + seed apo-DnaA, so the
    DnaA level is the physiological ~325 (Stage-1) rather than v2ecoli's
    ~12×-high ParCa default. Use with cache_dir=out/cache-stage1-heuristic
    (which zeroes ParCa DnaA translation). This is how dnaa-02 gets the
    corrected DnaA level the expert asked for; the intrinsic-hydrolysis +
    equilibrium then act on that level.

    Two mechanisms are supported:

    - ``mechanism='rida-v0'`` (default, canonical per dnaa-02f):
      The equilibrium reaction set stays INTACT. A DnaaRidaV0 Step is
      added between intrinsic hydrolysis and the clamp; when ≥1
      replisome is active it supplies an extra ATP→ADP flux at
      ``rida_rate_per_min`` (default 4.6 = literature 100× Sekimizu
      intrinsic). The clamp + RIDA together pin atp_fraction at the
      upper band edge against the equilibrium's reverse flux. No
      monkey-patch, no fictional species.

    - ``mechanism='monkey-patch'`` (legacy, 2026-05-17 option-B):
      _disable_dnaa_adp_equilibrium mutates the live ecoli-equilibrium
      Step's stoichMatrix to zero out the MONOMER0-4565_RXN column.
      Kept for backward compatibility + reproducibility of the
      original dnaa-02 verdict. dnaa-02f superseded this.

    Args:
        core: bigraph-schema core. If None, baseline() creates one.
        seed: RNG seed for stochastic init.
        cache_dir: ParCa cache directory.
        mechanism: 'rida-v0' (default) or 'monkey-patch' (legacy).
        rida_rate_per_min: RIDA-v0 effective rate (rida-v0 only).
        hydrolysis_rate_per_min: intrinsic DnaA-ATP→DnaA-ADP rate
            (Sekimizu 1987 default 0.046/min).
        hydrolysis_deterministic: round expected transfers (tests).
        atp_fraction_clamp_low, atp_fraction_clamp_high: if both set,
            the clamp Step holds DnaA-ATP / total inside [low, high].
            Both ``None`` disables the clamp.

    Returns:
        process-bigraph state document.
    """
    if mechanism not in ("intrinsic", "rida-v0", "monkey-patch"):
        raise ValueError(
            f"dnaa_02_with_intrinsic_hydrolysis: mechanism must be "
            f"'intrinsic', 'rida-v0' or 'monkey-patch', got {mechanism!r}"
        )
    # 'intrinsic' (expert-preferred for dnaa-02, round-1 feedback 2026-05-21):
    # equilibrium INTACT, NO RIDA step. RIDA / DDAH / DARS are extrinsic
    # regulators whose loci/partners (datA, DARS1/2) aren't modelled yet, so
    # adding RIDA here is premature — the reviewer asked for intrinsic
    # hydrolysis ONLY at this stage. Pair with atp_fraction_clamp_*=None to
    # see the honest intrinsic-only behaviour (no artificial band-forcing).
    # 'rida-v0' (former dnaa-02f default) is retained for the later study
    # where RIDA's prerequisites exist; 'monkey-patch' is the legacy
    # equilibrium-zeroing option.

    # 1. Build the standard baseline composite.
    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    # 1b. Biology mechanism — RIDA-v0 (default) leaves the equilibrium
    #     intact; monkey-patch (legacy) zeros MONOMER0-4565_RXN.
    if mechanism == "monkey-patch":
        _disable_dnaa_adp_equilibrium(cell_state)

    # 1c. Stage-1 constitutive DnaA expression (opt-in): seed apo-DnaA + add
    #     the constitutive Step so DnaA sits at the physiological ~325 (with
    #     cache_dir=out/cache-stage1-heuristic zeroing ParCa DnaA translation).
    if stage1_dnaa_expression:
        from v2ecoli.steps.dnaa_constitutive_expression import DnaaConstitutiveExpression
        from v2ecoli.library.schema import bulk_name_to_idx
        if initial_dnaa_count and "bulk" in cell_state:
            try:
                _ai = int(bulk_name_to_idx("PD03831[c]", cell_state["bulk"]["id"]))
                cell_state["bulk"]["count"][_ai] = int(initial_dnaa_count)
            except Exception:
                pass
        _expr = DnaaConstitutiveExpression(parameters={
            "rate_protein_per_min_per_gene": float(dnaa_synthesis_rate_per_min),
            "seed": int(seed),
        })
        _append_step(cell_state, "dnaa-constitutive-expression",
                     _expr, _PRIORITY_HYDROLYSIS)

    # 2. Instantiate Steps.
    hydrolysis = DnaaIntrinsicHydrolysis(parameters={
        "rate_per_min":   float(hydrolysis_rate_per_min),
        "deterministic":  bool(hydrolysis_deterministic),
        "seed":           int(seed),
    })

    rida = None
    if mechanism == "rida-v0":
        rida = DnaaRidaV0(parameters={
            "rida_rate_per_min":      float(rida_rate_per_min),
            "intrinsic_rate_per_min": float(hydrolysis_rate_per_min),
            "deterministic":          False,
            "seed":                   int(seed),
        })

    clamp_band = None
    if (atp_fraction_clamp_low is not None
            and atp_fraction_clamp_high is not None):
        clamp_band = [float(atp_fraction_clamp_low),
                      float(atp_fraction_clamp_high)]
    clamp = DnaaAtpFractionClamp(parameters={"band": clamp_band})

    listener = DnaaCycleListener(parameters={})

    # 3. Token-chained Step ordering: intrinsic → [RIDA] → clamp → listener.
    cell_state.setdefault("_dnaa02_token_h", 0.0)
    if rida is not None:
        cell_state.setdefault("_dnaa02_token_r", 0.0)
    cell_state.setdefault("_dnaa02_token_c", 0.0)

    _append_step(cell_state, "dnaa-intrinsic-hydrolysis",
                 hydrolysis, _PRIORITY_HYDROLYSIS,
                 token_out="_dnaa02_token_h")
    if rida is not None:
        _append_step(cell_state, "dnaa-rida-v0",
                     rida, _PRIORITY_RIDA,
                     token_in="_dnaa02_token_h",
                     token_out="_dnaa02_token_r")
        _append_step(cell_state, "dnaa-atp-fraction-clamp",
                     clamp, _PRIORITY_CLAMP,
                     token_in="_dnaa02_token_r",
                     token_out="_dnaa02_token_c")
    else:
        _append_step(cell_state, "dnaa-atp-fraction-clamp",
                     clamp, _PRIORITY_CLAMP,
                     token_in="_dnaa02_token_h",
                     token_out="_dnaa02_token_c")
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER,
                 token_in="_dnaa02_token_c")

    # 4. Extend flow_order so listing tools include the new Steps.
    flow = []
    if stage1_dnaa_expression:
        flow.append("dnaa-constitutive-expression")
    flow.append("dnaa-intrinsic-hydrolysis")
    if rida is not None:
        flow.append("dnaa-rida-v0")
    flow.extend(["dnaa-atp-fraction-clamp", "dnaa-cycle-listener"])
    doc["flow_order"] = list(doc.get("flow_order", [])) + flow

    return doc


# ─── dnaa-00: baseline + DnaA-cycle readout (observability only) ───────────
@composite_generator(
    name="dnaa_00_baseline_with_dnaa_readout",
    description=(
        "v2ecoli default whole-cell baseline + the read-only DnaaCycleListener. "
        "Adds NO mechanism and changes NO dynamics — it only computes the DnaA "
        "nucleotide-state readout (apo / DnaA-ATP / DnaA-ADP counts + fractions) "
        "from the existing bulk molecules PD03831[c], MONOMER0-160[c], "
        "MONOMER0-4565[c] each tick, so listeners.dnaA_cycle.atp_fraction is "
        "available on the plain baseline. Used as dnaa-00's baseline run so the "
        "baseline-vs-Stage-1 comparatives can overlay an atp_fraction trace for "
        "the unmodified cell."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0, "description": "RNG seed"},
        "cache_dir": {"type": "string",  "default": "out/cache",
                      "description": "Path to ParCa cache directory"},
    },
)
@composite_generator(
    name="dnaa_stage1_constitutive",
    description=(
        "Stage-1 dnaA expression (expert round-1, 2026-05-21): a constitutive "
        "apo-DnaA source at 1.5 protein/min/gene (= transcription 1.5 × TE 1), "
        "+ the read-only dnaA-cycle listener. Pair with the 'stage1-heuristic' "
        "condition cache (cache_dir=out/cache-stage1-heuristic), which zeroes "
        "ParCa's DnaA translation so this Step is the SOLE DnaA source. "
        "Replaces the non-convergent ParCa-patch approach (Option B). "
        "Constitutive = no autorepression yet, by the expert's staging."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0, "description": "RNG seed"},
        "cache_dir": {"type": "string",  "default": "out/cache-stage1-heuristic",
                      "description": "Stage-1 condition cache (TE[DnaA]=0)"},
        "rate_protein_per_min_per_gene": {
            "type": "number", "default": 1.5,
            "description": "Constitutive DnaA synthesis rate (Stage-1: 1.5)"},
        "gene_copies": {
            "type": "number", "default": 1.0,
            "description": "dnaA gene-copy multiplier (slow non-overlapping cycle ≈ 1)"},
        "initial_dnaa_count": {
            "type": "integer", "default": 325,
            "description": "Seed apo-DnaA count. The stage1 cache zeroes ParCa "
                           "DnaA translation, so without seeding the cell starts "
                           "at 0 DnaA (long unphysical transient). Default 325 ≈ "
                           "rate·τ/ln2 (steady state at 1.5/min, τ=150min) and in "
                           "the literature band 300-800 (Schmidt 2016)."},
    },
)
def dnaa_stage1_constitutive(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache-stage1-heuristic",
    rate_protein_per_min_per_gene: float = 1.5,
    gene_copies: float = 1.0,
    initial_dnaa_count: int = 325,
) -> dict:
    """Stage-1 dnaA expression: constitutive apo-DnaA source + cycle readout.

    Uses the 'stage1-heuristic' condition cache (ParCa DnaA translation
    zeroed) so the DnaaConstitutiveExpression Step is the only DnaA source,
    producing apo-DnaA at the absolute Stage-1 rate. See the Step docstring
    and feedback-2026-05-21/PLAN.md.
    """
    from v2ecoli.steps.dnaa_constitutive_expression import DnaaConstitutiveExpression
    from v2ecoli.library.schema import bulk_name_to_idx

    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    # Seed apo-DnaA: the stage1 cache zeroes ParCa DnaA translation, so the
    # initial proteome has 0 DnaA. Start at a physiological count (~325) so
    # the cell is at steady state from t=0; the equilibrium then partitions
    # apo into the DnaA-ATP/ADP forms (fast). Without this the run shows a
    # full-generation build-up from 0.
    if initial_dnaa_count and "bulk" in cell_state:
        bulk = cell_state["bulk"]
        try:
            apo_idx = int(bulk_name_to_idx("PD03831[c]", bulk["id"]))
            bulk["count"][apo_idx] = int(initial_dnaa_count)
        except Exception:
            pass

    expr = DnaaConstitutiveExpression(parameters={
        "rate_protein_per_min_per_gene": float(rate_protein_per_min_per_gene),
        "gene_copies": float(gene_copies),
        "seed": int(seed),
    })
    _append_step(cell_state, "dnaa-constitutive-expression",
                 expr, _PRIORITY_HYDROLYSIS)   # runs before the listener
    listener = DnaaCycleListener(parameters={})
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER)
    doc["flow_order"] = list(doc.get("flow_order", [])) + [
        "dnaa-constitutive-expression", "dnaa-cycle-listener"]
    return doc


def dnaa_00_baseline_with_dnaa_readout(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
) -> dict:
    """Baseline + read-only DnaaCycleListener (observability only).

    The DnaA-ATP / DnaA-ADP / apo-DnaA species exist in the WCM baseline as
    standard bulk molecules; the plain baseline composite simply doesn't wire
    in a Step to compute their fractions. This recipe adds ONLY that listener
    (a pure observer — reads bulk, writes listeners.dnaA_cycle.*, touches no
    dynamics), so the default cell's DnaA-ATP fraction is visible without
    altering its behaviour. See the dnaa-cycle-listener Step docstring.
    """
    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    listener = DnaaCycleListener(parameters={})
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER)
    doc["flow_order"] = list(doc.get("flow_order", [])) + ["dnaa-cycle-listener"]
    return doc


# ─── dnaa-03: box binding ──────────────────────────────────────────────────
@composite_generator(
    name="dnaa_03_with_box_binding",
    description=(
        "v2ecoli baseline + dnaa-02 nucleotide cycle + dnaa-03 DnaA-box "
        "binding (322+ catalogued sites: 11 oriC + 7 dnaAp + 307 chromosomal "
        "+ datA/DARS placeholders). Inherits dnaa-02's mechanism param "
        "(default 'rida-v0' per dnaa-02f)."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0},
        "cache_dir": {"type": "string",  "default": "out/cache"},
        # dnaa-02 params (inherited)
        "mechanism": {"type": "string",  "default": "rida-v0"},
        "rida_rate_per_min": {"type": "number", "default": 4.6},
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
    mechanism: str = "rida-v0",
    rida_rate_per_min: float = 4.6,
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

    Stacks on top of dnaa-02 (intrinsic hydrolysis + ATP-fraction clamp +
    the dnaa-02 nucleotide-cycle mechanism — 'rida-v0' by default after
    dnaa-02f, or 'monkey-patch' for legacy reproduction).

    The dnaa-02 default clamp band [0.2, 0.5] is on by default here so
    the box-binding model sees a physiologically reasonable ATP/ADP
    free-pool split. Override with ``atp_fraction_clamp_low=None``
    if you want the intrinsic-only behavior.
    """
    doc = dnaa_02_with_intrinsic_hydrolysis(
        core=core, seed=seed, cache_dir=cache_dir,
        mechanism=mechanism,
        rida_rate_per_min=rida_rate_per_min,
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


# ─── dnaa-02f variant E: RIDA-v0 placeholder ──────────────────────────────
@composite_generator(
    name="dnaa_02f_with_rida_v0",
    description=(
        "v2ecoli baseline + dnaa-02 intrinsic hydrolysis + RIDA-v0 "
        "(fork-active 100× hydrolysis multiplier). Variant E of the "
        "dnaa-02f equilibrium-cleanup follow-up. Equilibrium reactions "
        "stay INTACT — no stoichMatrix mutation, no fictional species."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0},
        "cache_dir": {"type": "string",  "default": "out/cache"},
        "hydrolysis_rate_per_min": {
            "type": "number", "default": 0.046,
            "description": "Sekimizu 1987 intrinsic rate (constitutive).",
        },
        "rida_rate_per_min": {
            "type": "number", "default": 4.6,
            "description": "RIDA-v0 effective rate when ≥1 replisome is "
                           "active (default 100× intrinsic, order-of-"
                           "magnitude literature estimate for S phase).",
        },
        "atp_fraction_clamp_low":  {"type": "number", "default": None},
        "atp_fraction_clamp_high": {"type": "number", "default": None},
    },
)
def dnaa_02f_with_rida_v0(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    hydrolysis_rate_per_min: float = 0.046,
    rida_rate_per_min: float = 4.6,
    atp_fraction_clamp_low: float | None = None,
    atp_fraction_clamp_high: float | None = None,
) -> dict:
    """Build the dnaa-02f variant-E composite.

    Differs from ``dnaa_02_with_intrinsic_hydrolysis`` in two ways:

    1. **Does NOT call ``_disable_dnaa_adp_equilibrium``.** MONOMER0-4565_RXN
       stays in the equilibrium reaction set. This is the explicit test:
       can RIDA's 100× scaling outpace the equilibrium's recovery?
    2. **Adds a DnaaRidaV0 Step** between intrinsic-hydrolysis and the
       clamp. RIDA only acts during S phase (when at least one replisome
       is in unique.active_replisome).

    Args:
        core: bigraph-schema core. If None, baseline() creates one.
        seed: RNG seed.
        cache_dir: ParCa cache directory.
        hydrolysis_rate_per_min: constitutive intrinsic rate (Sekimizu 1987).
        rida_rate_per_min: S-phase effective rate (RIDA-v0).
        atp_fraction_clamp_low, atp_fraction_clamp_high: optional clamp
            band; both None disables.

    Returns:
        process-bigraph state document.
    """
    # 1. Baseline composite. NO monkey-patch — equilibrium stays intact.
    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    # 2. Steps — intrinsic hydrolysis, RIDA-v0, clamp, listener.
    hydrolysis = DnaaIntrinsicHydrolysis(parameters={
        "rate_per_min":  float(hydrolysis_rate_per_min),
        "deterministic": False,
        "seed":          int(seed),
    })
    rida = DnaaRidaV0(parameters={
        "rida_rate_per_min":      float(rida_rate_per_min),
        "intrinsic_rate_per_min": float(hydrolysis_rate_per_min),
        "deterministic":          False,
        "seed":                   int(seed),
    })

    clamp_band = None
    if (atp_fraction_clamp_low is not None
            and atp_fraction_clamp_high is not None):
        clamp_band = [float(atp_fraction_clamp_low),
                      float(atp_fraction_clamp_high)]
    clamp = DnaaAtpFractionClamp(parameters={"band": clamp_band})

    listener = DnaaCycleListener(parameters={})

    # 3. Token chain: intrinsic → rida → clamp → listener.
    cell_state.setdefault("_dnaa02f_token_h",  0.0)
    cell_state.setdefault("_dnaa02f_token_r",  0.0)
    cell_state.setdefault("_dnaa02f_token_c",  0.0)

    _append_step(cell_state, "dnaa-intrinsic-hydrolysis",
                 hydrolysis, _PRIORITY_HYDROLYSIS,
                 token_out="_dnaa02f_token_h")
    _append_step(cell_state, "dnaa-rida-v0",
                 rida, _PRIORITY_RIDA,
                 token_in="_dnaa02f_token_h",
                 token_out="_dnaa02f_token_r")
    _append_step(cell_state, "dnaa-atp-fraction-clamp",
                 clamp, _PRIORITY_CLAMP,
                 token_in="_dnaa02f_token_r",
                 token_out="_dnaa02f_token_c")
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER,
                 token_in="_dnaa02f_token_c")

    doc["flow_order"] = list(doc.get("flow_order", [])) + [
        "dnaa-intrinsic-hydrolysis",
        "dnaa-rida-v0",
        "dnaa-atp-fraction-clamp",
        "dnaa-cycle-listener",
    ]

    return doc


# ─── dnaa-02f variant B': recalibrated equilibrium reverse rate ────────────
@composite_generator(
    name="dnaa_02f_with_recalibrated_equilibrium",
    description=(
        "v2ecoli baseline + dnaa-02 intrinsic hydrolysis + scaled "
        "MONOMER0-4565_RXN equilibrium reaction. Variant B' of dnaa-02f: "
        "tests whether the v2ecoli equilibrium reverse-rate is over-"
        "estimated. ``dnaa_adp_rxn_reverse_rate_scale`` scales the "
        "reaction's stoichMatrix column (1.0 = intact, 0.0 = same as "
        "option B). No fictional species — operates within the real "
        "reaction set."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0},
        "cache_dir": {"type": "string",  "default": "out/cache"},
        "hydrolysis_rate_per_min": {
            "type": "number", "default": 0.046,
            "description": "Sekimizu 1987 intrinsic rate (constitutive).",
        },
        "dnaa_adp_rxn_reverse_rate_scale": {
            "type": "number", "default": 0.01,
            "description": "Scale factor on MONOMER0-4565_RXN's stoichMatrix "
                           "column. 1.0 leaves the equilibrium intact; 0.0 "
                           "matches the option-B disable. Default 0.01 = 100× "
                           "attenuation, a starting heuristic.",
        },
        "atp_fraction_clamp_low":  {"type": "number", "default": None},
        "atp_fraction_clamp_high": {"type": "number", "default": None},
    },
)
def dnaa_02f_with_recalibrated_equilibrium(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    hydrolysis_rate_per_min: float = 0.046,
    dnaa_adp_rxn_reverse_rate_scale: float = 0.01,
    atp_fraction_clamp_low: float | None = None,
    atp_fraction_clamp_high: float | None = None,
) -> dict:
    """Build the dnaa-02f variant-B' composite.

    Differs from ``dnaa_02_with_intrinsic_hydrolysis`` by replacing the
    binary ``_disable_dnaa_adp_equilibrium`` patch with a continuous
    ``_scale_dnaa_adp_equilibrium(cell_state, scale)`` — same Step graph,
    just a parametric equilibrium-recovery rate. Lets us sweep the scale
    to find the threshold at which the equilibrium no longer dominates.

    No new bulk ids, no fork-gating, no fictional species — operates
    entirely within the existing real reaction set. The biological
    interpretation: ``dnaa_adp_rxn_reverse_rate_scale`` represents our
    best-guess correction factor for ParCa's fitted 2.9e-8 reverse rate
    (dnaa-02-EQ-01 open question).
    """
    doc = baseline(core=core, seed=seed, cache_dir=cache_dir)
    cell_state = doc["state"]["agents"]["0"]

    # The biology pivot: scale instead of disable.
    _scale_dnaa_adp_equilibrium(
        cell_state, scale=float(dnaa_adp_rxn_reverse_rate_scale))

    hydrolysis = DnaaIntrinsicHydrolysis(parameters={
        "rate_per_min":  float(hydrolysis_rate_per_min),
        "deterministic": False,
        "seed":          int(seed),
    })

    clamp_band = None
    if (atp_fraction_clamp_low is not None
            and atp_fraction_clamp_high is not None):
        clamp_band = [float(atp_fraction_clamp_low),
                      float(atp_fraction_clamp_high)]
    clamp = DnaaAtpFractionClamp(parameters={"band": clamp_band})

    listener = DnaaCycleListener(parameters={})

    cell_state.setdefault("_dnaa02bp_token_h", 0.0)
    cell_state.setdefault("_dnaa02bp_token_c", 0.0)

    _append_step(cell_state, "dnaa-intrinsic-hydrolysis",
                 hydrolysis, _PRIORITY_HYDROLYSIS,
                 token_out="_dnaa02bp_token_h")
    _append_step(cell_state, "dnaa-atp-fraction-clamp",
                 clamp, _PRIORITY_CLAMP,
                 token_in="_dnaa02bp_token_h",
                 token_out="_dnaa02bp_token_c")
    _append_step(cell_state, "dnaa-cycle-listener",
                 listener, _PRIORITY_LISTENER,
                 token_in="_dnaa02bp_token_c")

    doc["flow_order"] = list(doc.get("flow_order", [])) + [
        "dnaa-intrinsic-hydrolysis",
        "dnaa-atp-fraction-clamp",
        "dnaa-cycle-listener",
    ]

    return doc


# ─── dnaa-04: initiation-mechanism shadow observer ─────────────────────────
@composite_generator(
    name="dnaa_04_with_dnaa_initiation_trigger",
    description=(
        "v2ecoli baseline + dnaa-02 + dnaa-03 + DnaaInitiationMechanism "
        "shadow observer. The observer emits a 'would_fire' signal each "
        "tick when (oriC_high ≥ θ_oric AND atp_fraction ≥ θ_atp AND "
        "not in SeqA-v0 refractory). Does NOT replace v2ecoli's actual "
        "initiation Process — it runs in parallel and provides observables "
        "for comparing the mechanistic trigger against the existing "
        "heuristic. dnaa-04 first-pass scope."
    ),
    parameters={
        "seed":      {"type": "integer", "default": 0},
        "cache_dir": {"type": "string",  "default": "out/cache"},
        # Inherited from dnaa-02 / dnaa-03
        "mechanism":         {"type": "string",  "default": "rida-v0"},
        "rida_rate_per_min": {"type": "number",  "default": 4.6},
        "hydrolysis_rate_per_min":  {"type": "number", "default": 0.046},
        "atp_fraction_clamp_low":   {"type": "number", "default": 0.2},
        "atp_fraction_clamp_high":  {"type": "number", "default": 0.5},
        "kd_high_nM":           {"type": "number",  "default": 1.0},
        "kd_low_nM":            {"type": "number",  "default": 100.0},
        "enable_oric_binding":  {"type": "boolean", "default": True},
        "enable_dnaap_binding": {"type": "boolean", "default": True},
        "initial_dnaA_count_per_cell": {"type": "integer", "default": None},
        # dnaa-04 specific
        "oric_high_threshold":     {"type": "number", "default": 0.8,
                                    "description": "oriC high-affinity occupancy threshold for would_fire."},
        "atp_fraction_threshold":  {"type": "number", "default": 0.3,
                                    "description": "DnaA-ATP / total threshold for would_fire."},
        "refractory_seconds":      {"type": "number", "default": 600.0,
                                    "description": "SeqA-v0 fixed-timer refractory after each would_fire event."},
    },
)
def dnaa_04_with_dnaa_initiation_trigger(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    mechanism: str = "rida-v0",
    rida_rate_per_min: float = 4.6,
    hydrolysis_rate_per_min: float = 0.046,
    atp_fraction_clamp_low: float | None = 0.2,
    atp_fraction_clamp_high: float | None = 0.5,
    kd_high_nM: float = 1.0,
    kd_low_nM: float = 100.0,
    enable_oric_binding: bool = True,
    enable_dnaap_binding: bool = True,
    initial_dnaA_count_per_cell: int | None = None,
    oric_high_threshold: float = 0.8,
    atp_fraction_threshold: float = 0.3,
    refractory_seconds: float = 600.0,
) -> dict:
    """Build the dnaa-04 shadow-observer composite.

    Stacks dnaa-03 (which itself stacks dnaa-02) and appends a single
    Step: DnaaInitiationMechanism. The Step reads
    ``listeners.dnaA_binding.oric_high_occupied`` and
    ``listeners.dnaA_cycle.atp_fraction`` per tick, applies the trigger
    + SeqA-v0 refractory logic, and emits to
    ``listeners.dnaA_initiation.{would_fire, oric_high_obs, atp_fraction_obs,
    in_refractory, t_since_last_fire_s, cumulative_fires}``.

    The would_fire signal is purely observational. Future passes will
    replace v2ecoli's existing mass-threshold heuristic with this
    trigger (requires direct edit of the chromosome process).
    """
    doc = dnaa_03_with_box_binding(
        core=core, seed=seed, cache_dir=cache_dir,
        mechanism=mechanism,
        rida_rate_per_min=rida_rate_per_min,
        hydrolysis_rate_per_min=hydrolysis_rate_per_min,
        atp_fraction_clamp_low=atp_fraction_clamp_low,
        atp_fraction_clamp_high=atp_fraction_clamp_high,
        kd_high_nM=kd_high_nM,
        kd_low_nM=kd_low_nM,
        enable_oric_binding=enable_oric_binding,
        enable_dnaap_binding=enable_dnaap_binding,
        initial_dnaA_count_per_cell=initial_dnaA_count_per_cell,
    )
    cell_state = doc["state"]["agents"]["0"]

    initiation = DnaaInitiationMechanism(parameters={
        "oric_high_threshold":    float(oric_high_threshold),
        "atp_fraction_threshold": float(atp_fraction_threshold),
        "refractory_seconds":     float(refractory_seconds),
    })

    _append_step(cell_state, "dnaa-initiation-mechanism",
                 initiation, _PRIORITY_INITIATION)

    doc["flow_order"] = list(doc.get("flow_order", [])) + [
        "dnaa-initiation-mechanism",
    ]

    return doc
