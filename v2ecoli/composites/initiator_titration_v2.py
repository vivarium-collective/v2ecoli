"""Initiator-titration model v2 — standalone composite recipe.

Mean-field model of *E. coli* replication initiation from
Fu, Xiao & Jun (PRX Life 2023). See
``references/notes/FuXiaoJun2023.md`` for the paper digest and
``v2ecoli/processes/initiator_titration_v2.py`` for the Process body.

This composite is **standalone**: it wires ONE Step (the ITv2 mean-field
model) into a minimal bigraph and emits its observables. Nothing from
v2ecoli's whole-cell machinery is required — no ParCa cache load, no
bulk-molecule store, no listener stack. Total tick cost ~10 µs.

Why a separate composite (not a recipe over baseline.py)? Because the
purpose of this model is to provide a **comparison target** for v2ecoli's
much richer DnaA mechanism. Running it side-by-side keeps the comparison
apples-to-apples at the observable level (initiation mass, DnaA-ATP
fraction, cell-cycle timing) without v2ecoli's machinery contaminating
the FXJ trace.

Variants registered:
  - ``initiator_titration_v2``         canonical wild-type parameters (with RIDA)
  - ``initiator_titration_v2_delta4``  Δ4 mutant (RIDA + DARS + DDAH disabled);
                                       shows oscillatory initiation instability
                                       in fast-growth regimes per the paper.
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import make_edge
from v2ecoli.processes.initiator_titration_v2 import InitiatorTitrationV2


_PARAMS = {
    "tau_min": {
        "type": "number", "default": 60.0,
        "description": "Mass-doubling time τ in minutes. Sets λ = ln(2)/τ.",
    },
    "C_min": {
        "type": "number", "default": 40.0,
        "description": "Replication-period C in minutes (ori → ter traversal time).",
    },
    "D_min": {
        "type": "number", "default": 20.0,
        "description": "Termination → division delay D in minutes.",
    },
    "N_B": {
        "type": "integer", "default": 300,
        "description": "Number of high-affinity chromosomal DnaA boxes.",
    },
    "n_B": {
        "type": "integer", "default": 10,
        "description": "Number of low-affinity DnaA boxes inside oriC.",
    },
    "c_I_per_um3": {
        "type": "number", "default": 600.0,
        "description": "Target DnaA concentration in copies / µm³ (≈ Schmidt 2016).",
    },
    "nu_intrinsic_per_min": {
        "type": "number", "default": 0.046,
        "description": "DnaA-ATP → DnaA-ADP intrinsic ATPase rate (per minute).",
    },
    "nu_rida_per_min": {
        "type": "number", "default": 0.20,
        "description": (
            "Additional RIDA hydrolysis rate when ≥1 fork is active. "
            "Set to 0 for the Δ4-mutant condition."
        ),
    },
    "eclipse_s": {
        "type": "number", "default": 600.0,
        "description": "Eclipse period (post-initiation refractory) in seconds.",
    },
    "V0_um3": {
        "type": "number", "default": 1.0,
        "description": "Initial cell volume (µm³).",
    },
    "I_T0": {
        "type": "number", "default": 500.0,
        "description": "Initial DnaA-ATP count.",
    },
    "I_D0": {
        "type": "number", "default": 100.0,
        "description": "Initial DnaA-ADP count.",
    },
    "divide_on_terminate": {
        "type": "boolean", "default": True,
        "description": (
            "Halve V / I_T / I_D D minutes after each termination. "
            "False = protocell mode (no division, unbounded growth)."
        ),
    },
    "substep_s": {
        "type": "number", "default": 1.0,
        "description": "Internal Euler substep (seconds).",
    },
    "time_step": {
        "type": "number", "default": 60.0,
        "description": "Outer composite-tick interval (seconds).",
    },
    "seed": {
        "type": "integer", "default": 0,
        "description": "RNG seed (unused by the mean-field model; kept for API parity).",
    },
}


def _make_doc(process_config: dict) -> dict:
    """Build the bigraph doc around one InitiatorTitrationV2 instance.

    The Step writes its outputs to ``listeners.itv2.*`` — the same
    topology declared on the class. We seed those stores with their
    default values so the doc validates before the first tick.
    """
    instance = InitiatorTitrationV2(parameters=process_config)
    edge = make_edge(instance, instance.topology, edge_type='process')
    # Run the Process at the declared tick interval.
    edge['interval'] = float(process_config.get('time_step', 60.0))
    state = {
        "itv2": edge,
        "listeners": {
            "itv2": {
                "volume": 0.0,
                "dnaa_atp_count": 0.0,
                "dnaa_adp_count": 0.0,
                "dnaa_total_count": 0.0,
                "dnaa_atp_fraction": 0.0,
                "binding_sites": 0.0,
                "n_generations": 0,
                "fork_progress": [],
                "initiation_event": 0,
                "initiation_mass": 0.0,
                "initiations_so_far": 0,
                "divisions_so_far": 0,
            },
        },
    }
    # Wrap in the structured doc shape that process_bigraph.Composite
    # expects (matches v2ecoli/composites/baseline.py).
    return {
        "state": state,
        "skip_initial_steps": True,
        "sequential_steps": False,
        "flow_order": ["itv2"],
    }


@composite_generator(
    name="initiator_titration_v2",
    description=(
        "Standalone mean-field initiator-titration model v2 (Fu, Xiao, Jun "
        "PRX Life 2023). Tracks DnaA-ATP / DnaA-ADP, cell volume, and "
        "replication-fork progress without resolving individual DnaA boxes "
        "or other biomolecules. Wild-type defaults with RIDA stabilising "
        "the cycle. Run side-by-side with the v2ecoli baseline composite "
        "for an apples-to-apples comparison of initiation mass + cell-cycle "
        "timing."
    ),
    parameters=_PARAMS,
)
def initiator_titration_v2(core: Any = None, **kwargs) -> dict:
    """Build the bigraph doc for the canonical (wild-type) ITv2 run."""
    config = {k: kwargs.get(k, v["default"]) for k, v in _PARAMS.items()}
    return _make_doc(config)


_DELTA4_PARAMS = {**_PARAMS, "nu_rida_per_min": {
    "type": "number", "default": 0.0,
    "description": "Δ4 default: RIDA disabled. Override only if you want a partial Δ4.",
}}


@composite_generator(
    name="initiator_titration_v2_delta4",
    description=(
        "ITv2 in the Δ4-mutant condition: nu_rida_per_min = 0 by default. "
        "The paper predicts oscillatory initiation instability in fast-growth "
        "regimes; use this composite to reproduce that phenotype. Override "
        "tau_min to drive C/τ > 1."
    ),
    parameters=_DELTA4_PARAMS,
)
def initiator_titration_v2_delta4(core: Any = None, **kwargs) -> dict:
    """Build the bigraph doc for the Δ4-mutant ITv2 run."""
    config = {k: kwargs.get(k, v["default"]) for k, v in _DELTA4_PARAMS.items()}
    # Force the Δ4 condition (no RIDA) even if the user forgets the override.
    config["nu_rida_per_min"] = float(kwargs.get("nu_rida_per_min", 0.0))
    return _make_doc(config)
