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

import statistics
from v2ecoli.library.quantity_helpers import fg_magnitude
from typing import Any

from process_bigraph.composite import SyncUpdate

from v2ecoli.steps.base import V2Step


# scale name -> human description of the result slice it consumes
ANALYSIS_SCALES: dict[str, str] = {
    "single": "one cell's timeseries",
    "multidaughter": "sister cells from one division",
    "multigeneration": "cells across a lineage's generations",
    "multiseed": "cells across seeds of one variant",
    "multivariant": "cells across all variants",
}

# analysis name -> AnalysisStep subclass. Populated by __init_subclass__ for any
# subclass that defines its own ``name``; analysis_options entries resolve here.
ANALYSIS_REGISTRY: dict[str, type] = {}


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
        # Register concrete analyses (those declaring their own ``name``).
        if "name" in cls.__dict__:
            ANALYSIS_REGISTRY[cls.name] = cls

    def inputs(self):
        return {"results": "list"}

    def outputs(self):
        return {"analysis": "map"}

    def analyze(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def invoke(self, state, interval=None):
        # Analyses should fail loudly: unlike the simulation Steps (whose
        # V2Step.invoke swallows errors to keep the step cascade alive), a
        # broken or unimplemented analyze() must surface, not silently
        # return {}.
        return SyncUpdate(self.update(state))

    def update(self, state, interval=None):
        rows = state.get("results") or []
        return {"analysis": self.analyze(rows)}


class MassFractionSummary(AnalysisStep):
    """Single-scale example: mean mass fractions across a cell's timeseries."""

    name = "mass_fraction_summary"
    scale = "single"

    def analyze(self, rows):
        fraction_keys = ("protein", "rRna", "dna")
        empty = {"n_rows": 0, "n_valid_rows": 0,
                 **{f"{k}_fraction_mean": 0.0 for k in fraction_keys}}
        if not rows:
            return empty
        fractions = {k: [] for k in fraction_keys}
        for r in rows:
            mass = (r.get("listeners", {}) or {}).get("mass", {}) or {}
            dry = fg_magnitude(mass.get("dry_mass", 0.0)) or 0.0
            if dry <= 0:
                continue
            fractions["protein"].append(fg_magnitude(mass.get("protein_mass", 0.0)) / dry)
            fractions["rRna"].append(fg_magnitude(mass.get("rRna_mass", 0.0)) / dry)
            fractions["dna"].append(fg_magnitude(mass.get("dna_mass", 0.0)) / dry)
        n_valid = len(fractions["protein"])
        out = {"n_rows": len(rows), "n_valid_rows": n_valid}
        for name, vals in fractions.items():
            out[f"{name}_fraction_mean"] = (sum(vals) / len(vals)) if vals else 0.0
        return out


class DaughterMassSymmetry(AnalysisStep):
    """Multidaughter: birth-mass asymmetry |m1-m0|/(m1+m0) of sister cells.

    Dormant for single-lineage sweeps (only one daughter is carried); produces
    a value once binary-tree lineages (single_daughters=false) land.
    """

    name = "daughter_mass_symmetry"
    scale = "multidaughter"

    def analyze(self, rows):
        masses = [float(r.get("newborn_dry_mass", 0.0)) for r in rows]
        if len(masses) < 2:
            return {"n_sisters": len(masses),
                    "skipped": "needs >=2 daughters (single_daughters=false)"}
        m0, m1 = masses[0], masses[1]
        total = m0 + m1
        return {"n_sisters": len(masses),
                "mass_asymmetry": (abs(m1 - m0) / total) if total > 0 else 0.0}


class MassGrowthAcrossGenerations(AnalysisStep):
    """Multigeneration: per-generation newborn/final mass, cycle time, fold change
    across one lineage."""

    name = "mass_growth_across_generations"
    scale = "multigeneration"

    def analyze(self, rows):
        cells = sorted(rows, key=lambda r: int(r.get("generation", 0)))
        per_gen = []
        for c in cells:
            nb = float(c.get("newborn_dry_mass", 0.0))
            fn = float(c.get("final_dry_mass", 0.0))
            per_gen.append({
                "generation": int(c.get("generation", 0)),
                "newborn_dry_mass": nb, "final_dry_mass": fn,
                "division_time": float(c.get("division_time", 0.0)),
                "fold_change": (fn / nb) if nb > 0 else 0.0,
            })
        dts = [g["division_time"] for g in per_gen if g["division_time"] > 0]
        return {"n_generations": len(per_gen), "per_generation": per_gen,
                "mean_division_time": (sum(dts) / len(dts)) if dts else 0.0}


class DoublingTimeDistribution(AnalysisStep):
    """Multiseed: division-time mean/std over divided cells across seeds, plus
    mean final dry mass over all cells."""

    name = "doubling_time_distribution"
    scale = "multiseed"

    def analyze(self, rows):
        times = [float(r.get("division_time", 0.0)) for r in rows
                 # only confirmed divisions: a non-divided cell's division_time is
                 # the run cap, not a doubling time (divided None/False excluded)
                 if r.get("divided") is True and float(r.get("division_time", 0.0)) > 0]
        finals = [float(r.get("final_dry_mass", 0.0)) for r in rows
                  if float(r.get("final_dry_mass", 0.0)) > 0]
        return {
            "n_cells": len(rows),
            "n_divided": len(times),
            "doubling_time_mean": statistics.mean(times) if times else 0.0,
            "doubling_time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "final_dry_mass_mean": statistics.mean(finals) if finals else 0.0,
        }


class MetricAcrossVariants(AnalysisStep):
    """Multivariant: mean division time + final dry mass per variant."""

    name = "metric_across_variants"
    scale = "multivariant"

    def analyze(self, rows):
        by_variant: dict[int, list] = {}
        for r in rows:
            by_variant.setdefault(int(r.get("variant", 0)), []).append(r)
        per_variant = []
        for v, cells in sorted(by_variant.items()):
            dts = [float(c.get("division_time", 0.0)) for c in cells
                   # only confirmed divisions: a non-divided cell's division_time is
                   # the run cap, not a doubling time (divided None/False excluded)
                   if c.get("divided") is True and float(c.get("division_time", 0.0)) > 0]
            fms = [float(c.get("final_dry_mass", 0.0)) for c in cells
                   if float(c.get("final_dry_mass", 0.0)) > 0]
            per_variant.append({
                "variant": v,
                "n_cells": len(cells),
                "mean_division_time": statistics.mean(dts) if dts else 0.0,
                "mean_final_dry_mass": statistics.mean(fms) if fms else 0.0,
            })
        return {"per_variant": per_variant}


def _mean_std_cv(values: list[float]) -> dict[str, float]:
    """Ensemble central tendency + spread for a list of per-cell scalars.

    ``cv`` is the coefficient of variation (std / mean) — the dimensionless
    cell-to-cell spread that is the headline number for a growth-rate or
    composition grade. Undefined (0.0) when the mean is zero or n < 2.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0}
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "cv": (std / mean) if mean else 0.0}


class BasalPhenotypeCard(AnalysisStep):
    """Multiseed: the v1 *meta-tier report card* measurement.

    Grades the two universally-applicable basal-condition axes over an
    ensemble of cells (seeds x generations of one variant), reporting each
    as a cell-to-cell mean / std / CV:

      - **Growth**: doubling time over confirmed divisions.
      - **Composition**: protein / RNA / DNA mass fractions of dry weight.

    This is the *measurement* half of the report card. The *grade* (pass/fail
    vs. a pinned reference within tolerance) lives in
    ``tests/test_basal_phenotype_card.py`` so the reference and tolerances are
    reviewable artifacts, not buried analysis constants.

    Design notes:
      - ``generation_lower_bound`` drops early generations as burn-in, so the
        ensemble reflects steady balanced growth rather than the inoculation
        transient. The canonical convention is to discard the first few
        generations of a multi-gen lineage.
      - The per-cell records this consumes already carry time-averaged
        ``protein_fraction_mean`` / ``rRna_fraction_mean`` / ``dna_fraction_mean``
        (see ``analysis_runner.build_cell_records``), so composition is an
        ensemble reduction over those, not a re-derivation from timeseries.
      - **RNA == rRna for v1.** ``analysis_runner._MASS_COLS`` only pulls
        ``rRna_mass`` (rRna is ~80-85% of total RNA). Total-RNA/DW is a
        documented follow-up (extend ``_MASS_COLS`` with tRna/mRna).
    """

    name = "basal_phenotype_card"
    scale = "multiseed"
    config_schema = {
        "generation_lower_bound": {"_type": "integer", "_default": 0},
    }

    def analyze(self, rows):
        burn_in = int(self.config.get("generation_lower_bound", 0))
        kept = [r for r in rows if int(r.get("generation", 0)) >= burn_in]

        # Growth: doubling time over confirmed divisions only. A non-divided
        # cell's division_time is the run cap, not a doubling time.
        doubling_times = [
            float(r.get("division_time", 0.0)) for r in kept
            if r.get("divided") is True and float(r.get("division_time", 0.0)) > 0
        ]

        # Composition: ensemble reduction over the per-cell time-averaged
        # fractions. Skip cells with a zero/absent fraction (no valid mass).
        def _frac(key):
            return [float(r[key]) for r in kept
                    if float(r.get(key, 0.0)) > 0]

        return {
            "n_cells": len(kept),
            "generation_lower_bound": burn_in,
            "growth": {
                "doubling_time": _mean_std_cv(doubling_times),
            },
            "composition": {
                "protein_fraction": _mean_std_cv(_frac("protein_fraction_mean")),
                "rna_fraction": _mean_std_cv(_frac("rRna_fraction_mean")),
                "dna_fraction": _mean_std_cv(_frac("dna_fraction_mean")),
            },
        }
