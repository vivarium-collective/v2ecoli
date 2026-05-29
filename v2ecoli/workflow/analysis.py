"""Analyses as process-bigraph Steps that run on simulation results.

Defines the AnalysisStep base, the five-scale registry mirroring vEcoli's
analysis hierarchy, and one worked example (MassFractionSummary, ``single``
scale). Each scale declares which slice of emitted results it reads:

    single          one cell's timeseries
    multidaughter   sister cells from one division
    multigeneration cells across a lineage's generations
    multiseed       cells across seeds (same variant)
    multivariant    cells across all variants

Porting the full vEcoli analysis library onto this base is a follow-up spec.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.steps.base import V2Step


# scale name -> human description of the result slice it consumes
ANALYSIS_SCALES: dict[str, str] = {
    "single": "one cell's timeseries",
    "multidaughter": "sister cells from one division",
    "multigeneration": "cells across a lineage's generations",
    "multiseed": "cells across seeds of one variant",
    "multivariant": "cells across all variants",
}


class AnalysisStep(V2Step):
    """Base for result-consuming analysis Steps.

    Subclasses set ``scale`` (one of ANALYSIS_SCALES) and implement
    ``analyze(rows) -> dict``. ``rows`` is a list of emitted result records
    (dicts shaped like the partitioned parquet rows / in-state snapshots) for
    the slice this scale covers. The Step's update() reads ``results`` from
    state and writes the analysis output to ``analysis``.
    """

    scale: str = "single"
    config_schema = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.scale not in ANALYSIS_SCALES:
            raise ValueError(
                f"{cls.__name__}.scale={cls.scale!r} not in {sorted(ANALYSIS_SCALES)}")

    def inputs(self):
        return {"results": "list"}

    def outputs(self):
        return {"analysis": "map"}

    def analyze(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def update(self, state, interval=None):
        rows = state.get("results") or []
        return {"analysis": self.analyze(rows)}


class MassFractionSummary(AnalysisStep):
    """Single-scale example: mean mass fractions across a cell's timeseries."""

    name = "mass_fraction_summary"
    scale = "single"

    def analyze(self, rows):
        if not rows:
            return {"n_rows": 0}
        fractions = {"protein": [], "rRna": [], "dna": []}
        for r in rows:
            mass = (r.get("listeners", {}) or {}).get("mass", {}) or {}
            dry = float(mass.get("dry_mass", 0.0)) or 0.0
            if dry <= 0:
                continue
            fractions["protein"].append(float(mass.get("protein_mass", 0.0)) / dry)
            fractions["rRna"].append(float(mass.get("rRna_mass", 0.0)) / dry)
            fractions["dna"].append(float(mass.get("dna_mass", 0.0)) / dry)
        out: dict[str, Any] = {"n_rows": len(rows)}
        for name, vals in fractions.items():
            out[f"{name}_fraction_mean"] = (sum(vals) / len(vals)) if vals else 0.0
        return out
