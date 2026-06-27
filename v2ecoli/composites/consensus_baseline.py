"""Consensus elongation model composite — discoverable alias for the
kinetic_charging_baseline arch with both consensus flags forced on.

This composite IS the consensus elongation model from
``v2ecoli_consensus_model.md``:

* **Kinetic tRNA charging** with codon-aware reading (inherited from
  :mod:`v2ecoli.composites.kinetic_charging_baseline`, which swaps
  :class:`SteadyStatePolypeptideElongation` for
  :class:`KineticTrnaChargingPolypeptideElongation` in the partitioned-process
  registry).
* **AA synthesis / import / export** integrated INSIDE the same RK45 solve
  as tRNA charging (P3b-ii — ``include_aa_supply=True``).
* **ppGpp regulation** via RelA / SpoT-driven synthesis & degradation, plus
  pre-solve elongation-rate inhibition (P2 — ``ppgpp_regulation=True``).
* **aa_count_diff** feedback to metabolism's homeostatic FBA (P3b-ii tail).

Why a separate composite instead of just a config preset?

* **Discoverability**: ``consensus_baseline`` is the public name a user
  types when they want the consensus model. The fact that it shares 95%
  of its build with ``kinetic_charging_baseline`` is an implementation
  detail.
* **Default-on guarantee**: forcing the flags here means experiments
  bypassing ``config_overrides`` (notebooks, ad-hoc CLI runs, the
  registry-driven dashboard) can't accidentally land on the
  flags-off path and silently behave like the legacy kinetic model.
* **No physics duplication**: the body is a 3-line delegation to
  :func:`v2ecoli.composites.kinetic_charging_baseline.kinetic_charging_baseline`
  with the consensus flags merged into ``config_overrides``. Same RK45
  solve, same partitioner wiring, same listener emission.

If a user passes ``include_aa_supply=False`` or ``ppgpp_regulation=False``
via ``config_overrides``, their override wins — this composite enforces
the consensus defaults, not a hard lock.
"""

from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS
from v2ecoli.composites.kinetic_charging_baseline import (
    kinetic_charging_baseline as _kinetic_charging_baseline,
)


_CONSENSUS_DEFAULTS = {
    "ecoli-polypeptide-elongation.include_aa_supply": True,
    "ecoli-polypeptide-elongation.ppgpp_regulation": True,
}


@composite_generator(
    name="consensus_baseline",
    description=(
        "Consensus elongation model — kinetic_charging_baseline arch with "
        "include_aa_supply=True + ppgpp_regulation=True forced on. Couples "
        "kinetic tRNA charging, AA synthesis/import/export (inside the same "
        "ODE solve), and RelA/SpoT ppGpp regulation. See "
        "v2ecoli_consensus_model.md for the design and audit.md in "
        "workspace/investigations/consensus_elongation/ for the build."
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
            "description": "Path to ParCa cache directory",
        },
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": (
                "Declarative '<process>.<key>': value config overrides. "
                "User overrides win over the consensus defaults, so passing "
                "include_aa_supply=False or ppgpp_regulation=False here will "
                "degrade the consensus model to its constituent modes."
            ),
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    emitters=[
        {
            "address": "local:ParquetEmitter",
            "config": {},
            "paths": ["global_time", "bulk", "listeners"],
        },
    ],
)
def consensus_baseline(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    config_overrides: dict | None = None,
    bundle: dict | None = None,
) -> dict:
    """Build the process-bigraph state document for the consensus elongation
    model.

    Delegates to :func:`kinetic_charging_baseline` with the two consensus
    flags merged into ``config_overrides``. The user's overrides take
    precedence over the consensus defaults (a deliberate choice — lets
    you explore the degenerate modes without forking the composite).
    """
    merged = dict(_CONSENSUS_DEFAULTS)
    if config_overrides:
        merged.update(config_overrides)
    return _kinetic_charging_baseline(
        core=core,
        seed=seed,
        cache_dir=cache_dir,
        config_overrides=merged,
        bundle=bundle,
    )
