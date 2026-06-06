"""Pin a v1<->v2 *equivalence* reference for the basal population-phenotype card.

Unlike ``pin_population_phenotype_basal_reference.py`` (which self-pins v2ecoli
against its own blessed ensemble = a DRIFT instrument), this bakes a **vEcoli
("v1") ensemble** as the reference so the v2ecoli card grades as an EQUIVALENCE
instrument: "is v2 still the same E. coli as v1, within tolerance?"

It reuses the self-pin reference as the presentation/criterion TEMPLATE and only
swaps the ttest ``ref_values`` to the v1 ensemble's per-cell distributions.

Design note — **this does NOT touch shared v2ecoli infra.** The shared
``analysis_runner.build_cell_records`` expects the v2ecoli emit schema
(``global_time``; bulk as paired ``bulk__id``/``bulk__count``). vEcoli emits
``time`` (cumulative across the lineage) and positional bulk. Rather than make
the shared hot-path tolerant, we keep a small SELF-CONTAINED cross-implementation
reader here (``_read_vecoli_cells``) that handles those differences for the
Phase-1 axes (Physiology + Composition). Ribosomes/fluxes/omics (which need the
bulk-index adapter) are out of scope here and render absent => not graded.

Run:
    python scripts/pin_vecoli_equivalence_reference.py \\
        --sweep-dir /Users/chris/projects/SMS/vecoli-benchmarking/out/v1v2_smoke_2x2 \\
        --model-ref <vEcoli git SHA> \\
        --ensemble "2 seeds x 2 gens (smoke)" \\
        --out docs/report_cards/population_phenotype_basal/vs_vecoli/vecoli_reference.json

Layout: reference modes live as subdirs of the card they grade
(``<card>/vs_vecoli/``), so a card's self-pin (drift) and equivalence renderings
sit together. "Equivalence" is a reference MODE, not a card.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# axis path -> per-cell field this script computes. Phase-1 equivalence scope.
_AXES = {
    "physiology.doubling_time": "division_time",
    "physiology.cell_mass": "cell_mass_mean",
    "physiology.cell_volume": "volume_mean",
    "physiology.oric": "oric_mean",
    "physiology.replication_initiation": "replication_initiation_time",
    "physiology.replication_completion": "replication_completion_time",
    "composition.protein_fraction": "protein_fraction_mean",
    "composition.rna_fraction": "rna_fraction_mean",
    "composition.dna_fraction": "dna_fraction_mean",
}

_DEFAULT_TEMPLATE = "tests/fixtures/population_phenotype_basal_reference.json"


def _replication_events(times, oric, nforks):
    """First oriC step-up (initiation) and first fork-clear (completion), as
    times-since-birth. None if the event isn't observed in the cell's cycle.
    Mirrors analysis_runner._replication_events (kept local to avoid importing
    shared infra)."""
    init = next((times[i] for i in range(1, len(oric)) if oric[i] > oric[i - 1]),
                None)
    completion = next((times[i] for i in range(1, len(nforks))
                       if nforks[i] == 0 and nforks[i - 1] > 0), None)
    return init, completion


def _read_vecoli_cells(sweep_dir: str) -> list[dict]:
    """Cross-implementation reader for a vEcoli ("v1") parquet sweep, returning
    per-cell records in the same field shape the card's reference expects, for
    the Physiology + Composition axes only.

    Handles the two v1<->v2 schema differences (so shared infra stays untouched):
      * time column: vEcoli emits ``time`` (cumulative across the lineage), not
        ``global_time``; we normalize to since-birth so per-cell duration =
        last-first and replication events are since-birth, matching v2.
      * cell-level aggregation is identical to the card: time-mean WITHIN a cell
        (one value per cell); population stats live across cells downstream.
    """
    import duckdb

    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        return []
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    avail = {d[0] for d in duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet({flist}, hive_partitioning=true)"
    ).fetchall()}
    tcol = "global_time" if "global_time" in avail else "time"
    sel = (f"variant, lineage_seed, generation, agent_id, {tcol}, "
           "listeners__mass__dry_mass, listeners__mass__protein_mass, "
           "listeners__mass__rna_mass, listeners__mass__dna_mass, "
           "listeners__mass__cell_mass, listeners__mass__volume, "
           "listeners__replication_data__number_of_oric, "
           "len(listeners__replication_data__fork_coordinates)")
    rows = duckdb.sql(
        f"SELECT {sel} FROM read_parquet({flist}, hive_partitioning=true) "
        f"ORDER BY variant, lineage_seed, generation, agent_id, {tcol}"
    ).fetchall()

    by_cell: dict[tuple, list] = {}
    for (v, ls, g, a, t, dry, prot, rna, dna, cmass, vol, oric, nfork) in rows:
        by_cell.setdefault((int(v), int(ls), int(g), str(a)), []).append(
            (float(t), float(dry), float(prot), float(rna), float(dna),
             float(cmass), float(vol), float(oric), int(nfork)))

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    records = []
    for (v, ls, g, a), rs in by_cell.items():
        t0 = rs[0][0]
        times = [r[0] - t0 for r in rs]            # since-birth
        orics = [r[7] for r in rs]
        nforks = [r[8] for r in rs]
        prot_f = [r[2] / r[1] for r in rs if r[1] > 0]
        rna_f = [r[3] / r[1] for r in rs if r[1] > 0]
        dna_f = [r[4] / r[1] for r in rs if r[1] > 0]
        repl_init, repl_complete = _replication_events(times, orics, nforks)
        records.append({
            "variant": v, "lineage_seed": ls, "generation": g, "agent_id": a,
            "division_time": float(rs[-1][0] - rs[0][0]),
            "cell_mass_mean": mean([r[5] for r in rs]),
            "volume_mean": mean([r[6] for r in rs]),
            "oric_mean": mean(orics),
            "replication_initiation_time": repl_init,
            "replication_completion_time": repl_complete,
            "protein_fraction_mean": mean(prot_f),
            "rna_fraction_mean": mean(rna_f),
            "dna_fraction_mean": mean(dna_f),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", required=True, help="vEcoli parquet sweep dir")
    ap.add_argument("--template", default=_DEFAULT_TEMPLATE,
                    help="self-pin reference used as presentation/criterion template")
    ap.add_argument("--model-ref", required=True, help="vEcoli git commit (provenance)")
    ap.add_argument("--ensemble", default="", help="ensemble description (provenance)")
    ap.add_argument("--out", required=True, help="output reference json path")
    args = ap.parse_args()

    records = _read_vecoli_cells(args.sweep_dir)
    if not records:
        raise SystemExit(f"no cell records under {args.sweep_dir}")

    template = json.load(open(args.template))
    axes = {}
    print(f"v1 ensemble: {len(records)} cells")
    for path, field in _AXES.items():
        if path not in template["axes"]:
            continue
        vals = [r[field] for r in records if r.get(field) is not None]
        ax = dict(template["axes"][path])           # copy presentation + criterion
        ax["criterion"] = dict(ax["criterion"])
        ax["criterion"]["ref_values"] = vals        # swap in v1 distribution
        axes[path] = ax
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {path:36} n={len(vals):2}  mean={mean:.4g}")

    out = {
        "$schema_note": template.get("$schema_note", ""),
        "title": "Basal-condition population phenotype — v1↔v2 equivalence "
                 "(Physiology + Composition)",
        "status": "populated",
        "stimulus": {
            "reference_model": "vEcoli (v1)",
            "measured_model": "v2ecoli (v2)",
            "blessed_model_ref": args.model_ref,
            "ensemble": args.ensemble,
            "sweep_dir": args.sweep_dir,
        },
        "findings": [
            "EQUIVALENCE reference: v2ecoli graded against a vEcoli v1 ensemble "
            "(not a self-pin). Welch t-test of v2 cell-level values vs v1 ref_values.",
            "Phase-1 scope: Physiology + Composition. Ribosomes/fluxes/omics omitted "
            "(vEcoli emits bulk positionally, not as bulk__id/bulk__count; needs the "
            "bulk-index adapter).",
            "v1 values read by scripts/pin_vecoli_equivalence_reference.py "
            "(self-contained cross-impl reader; shared analysis_runner is untouched).",
            "tolerance bands (within_pct/mismatch_pct) inherited from the self-pin "
            "template; revisit per-axis equivalence margins (delta) for the real run.",
        ],
        "axes": axes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({len(axes)} axes)")


if __name__ == "__main__":
    main()
