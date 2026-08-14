"""Analyses as process-bigraph Steps that run on simulation results.

The ``AnalysisStep``/``Analysis`` bases + ``ANALYSIS_SCALES`` registry now live
in ``viva_superpowers.post_sim`` (the post-sim Step family's one home) and are
re-exported here for back-compat: existing call sites
(``from v2ecoli.workflow.analysis import AnalysisStep`` etc.) keep working
unchanged.

This module keeps the five-scale registry's concrete analyses mirroring
vEcoli's analysis hierarchy, starting with MassFractionSummary (``single``
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

from viva_superpowers.post_sim import (  # noqa: F401
    ANALYSIS_REGISTRY,
    ANALYSIS_SCALES,
    Analysis,
    AnalysisStep,
)


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
    """Population stats over a list of **per-cell** scalars (one value per cell).

    Carries the raw per-cell ``values`` so the renderer can draw a violin/strip
    and a Welch t-test can compare two ensembles directly from their value
    lists. ``cv`` is the coefficient of variation (std / mean) — the
    dimensionless cell-to-cell spread. ``std`` is the population standard
    deviation (descriptive); inferential comparisons use ``values`` directly.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "values": []}
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "cv": (std / mean) if mean else 0.0,
            "values": [float(v) for v in values]}


def _variance_decomposition(labeled: list[tuple]) -> dict:
    """Two-way (seed × generation) decomposition of the cell-to-cell variance.

    ``labeled`` is ``[(seed, generation, value), ...]``. Returns the fraction of
    total variance structured by seed (``eta_seed``) and by generation
    (``eta_gen``) — group-mean sums-of-squares over the grand total — plus a
    Spearman rank correlation of value vs generation (``rho_gen``). A high
    ``eta_gen`` that is also *monotonic* (|rho_gen| large) is the signature of a
    non-stationary lineage (the ensemble is still drifting across generations,
    not fluctuating around a steady state). Descriptive, not a formal ANOVA:
    the design is small + unbalanced, so eta's are marginal and need not sum to
    1. Returns ``{}`` when there's too little data or no variance."""
    n = len(labeled)
    if n < 6:
        return {}
    seeds = [s for s, _, _ in labeled]
    gens = [g for _, g, _ in labeled]
    xs = [v for _, _, v in labeled]
    mu = sum(xs) / n
    ss_tot = sum((x - mu) ** 2 for x in xs)
    if ss_tot == 0:
        return {}

    def _group_ss(keys):
        ss = 0.0
        for k in set(keys):
            idx = [i for i, kk in enumerate(keys) if kk == k]
            gm = sum(xs[i] for i in idx) / len(idx)
            ss += len(idx) * (gm - mu) ** 2
        return ss

    out = {"eta_seed": _group_ss(seeds) / ss_tot,
           "eta_gen": _group_ss(gens) / ss_tot,
           "n_seeds": len(set(seeds)), "n_gens": len(set(gens)),
           # raw labeled points so a flagged axis can plot metric-vs-generation
           "by_cell": [[int(s), int(g), float(v)] for s, g, v in labeled]}
    if len(set(gens)) > 1:
        from scipy import stats
        rho, p = stats.spearmanr(gens, xs)
        if rho == rho:  # not nan (constant values give nan)
            out["rho_gen"] = float(rho)
            out["p_gen"] = float(p)
    return out


def _stat(labeled: list[tuple]) -> dict:
    """Per-cell stat node ({n,mean,std,cv,values}) + a ``variance`` sub-dict
    decomposing the cell-to-cell variance across seeds vs generations.
    ``labeled`` is ``[(seed, generation, value), ...]``."""
    node = _mean_std_cv([v for _, _, v in labeled])
    dec = _variance_decomposition(labeled)
    if dec:
        node["variance"] = dec
    return node


class PopulationPhenotypeBasalCard(AnalysisStep):
    """Multiseed: the v1 *meta-tier report card* measurement.

    Grades the two universally-applicable basal-condition axes over an
    ensemble of cells (seeds x generations of one variant), reporting each
    as a cell-to-cell mean / std / CV:

      - **Growth**: doubling time over confirmed divisions.
      - **Composition**: protein / RNA / DNA mass fractions of dry weight.

    This is the *measurement* half of the report card. The *grade* (pass/fail
    vs. a pinned reference within tolerance) lives in
    ``tests/test_population_phenotype_basal.py`` so the reference and tolerances are
    reviewable artifacts, not buried analysis constants.

    Design notes:
      - ``generation_lower_bound`` drops early generations as burn-in, so the
        ensemble reflects steady balanced growth rather than the inoculation
        transient. The canonical convention is to discard the first few
        generations of a multi-gen lineage.
      - The per-cell records this consumes already carry time-averaged
        ``protein_fraction_mean`` / ``rna_fraction_mean`` / ``dna_fraction_mean``
        (see ``analysis_runner.build_cell_records``), so composition is an
        ensemble reduction over those, not a re-derivation from timeseries.
      - **RNA = total RNA** (``rna_mass`` = rRna+tRna+mRna), per cell.
      - Each axis carries the raw per-cell ``values`` (one per cell) so the
        report can draw violin/strip plots and grade with a Welch t-test.
    """

    name = "population_phenotype_basal"
    scale = "multiseed"
    config_schema = {
        "generation_lower_bound": {"_type": "integer", "_default": 0},
    }

    def analyze(self, rows):
        burn_in = int(self.config.get("generation_lower_bound", 0))
        kept = [r for r in rows if int(r.get("generation", 0)) >= burn_in]

        # Per-cell values carry their (seed, generation) labels so each axis can
        # decompose its cell-to-cell variance across seeds vs generations.
        def _lab(key, keep):
            out = []
            for r in kept:
                v = r.get(key)
                if v is None or not keep(r, float(v)):
                    continue
                out.append((int(r["lineage_seed"]), int(r["generation"]), float(v)))
            return out

        # doubling: confirmed divisions only (non-divided cells' time is the cap)
        _divided = lambda r, v: r.get("divided") is True and v > 0
        _pos = lambda r, v: v > 0          # fractions / levels: skip zero/absent
        _any = lambda r, v: True           # event times: already non-None

        # Simulation health: how the sims themselves went (divided vs hit the
        # per-generation duration cap), over ALL cells (pre-burn-in) — a
        # run-quality readout distinct from the graded phenotype axes. A
        # cell whose `divided` is False ran to the cap without dividing.
        n_div = sum(1 for r in rows if r.get("divided") is True)
        n_fail = sum(1 for r in rows if r.get("divided") is False)
        return {
            "n_cells": len(kept),
            "generation_lower_bound": burn_in,
            "sim_health": {"n_total": len(rows), "n_divided": n_div,
                           "n_failed": n_fail},
            "physiology": {
                "doubling_time": _stat(_lab("division_time", _divided)),
                "cell_mass": _stat(_lab("cell_mass_mean", _pos)),
                "cell_volume": _stat(_lab("volume_mean", _pos)),
                "oric": _stat(_lab("oric_mean", _pos)),
                "replication_initiation": _stat(_lab("replication_initiation_time", _any)),
                "replication_completion": _stat(_lab("replication_completion_time", _any)),
            },
            "composition": {
                "protein_fraction": _stat(_lab("protein_fraction_mean", _pos)),
                "rna_fraction": _stat(_lab("rna_fraction_mean", _pos)),
                "dna_fraction": _stat(_lab("dna_fraction_mean", _pos)),
            },
            "ribosomes": {
                "total": _stat(_lab("ribosome_total_mean", _pos)),
                "active_fraction": _stat(_lab("ribosome_active_fraction_mean", _pos)),
                "elongation_rate": _stat(_lab("ribosome_elongation_mean", _pos)),
                "production": _stat(_lab("ribosome_production_mean", _pos)),
            },
        }
