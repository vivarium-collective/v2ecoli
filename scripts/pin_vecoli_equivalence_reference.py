"""Pin a v1<->v2 *equivalence* reference for the basal population-phenotype card.

Unlike ``pin_population_phenotype_basal_reference.py`` (which self-pins v2ecoli
against its own blessed ensemble = a DRIFT instrument), this bakes a **vEcoli
("v1") ensemble** as the reference so the v2ecoli card grades as an EQUIVALENCE
instrument: "is v2 still the same E. coli as v1, within tolerance?"

It reuses the self-pin reference as the presentation/criterion TEMPLATE and only
swaps the reference values/vectors to the v1 ensemble's. **Full card**: Physiology,
Composition, Ribosomes (ttest), Exchange fluxes (scatter + KPIs), Gene expression
(r2). v1<->v2 share cistron/monomer/flux ordering exactly (verified), so vectors
align positionally — no ID remapping.

Design note — **this does NOT touch shared v2ecoli infra.** vEcoli's emit schema
differs from v2ecoli's in two ways, handled here in self-contained readers:
  * time: vEcoli emits ``time`` (cumulative across the lineage), not ``global_time``.
  * bulk: vEcoli emits bulk POSITIONALLY (one ``bulk`` list), not paired
    ``bulk__id``/``bulk__count`` — so the 30S/50S subunit counts are sliced by
    integer index (``--s30-idx``/``--s50-idx``, from v1 sim_data) for the
    ribosome total / active-fraction axes.
Omics/flux ensemble-mean vectors are read with the shared ``extract_vectors``
(column names match), since v1 emits the same listener columns.

Run (after extracting the two bulk indices from v1 sim_data in the vEcoli env):
    python scripts/pin_vecoli_equivalence_reference.py \\
        --sweep-dir /Users/chris/projects/SMS/vecoli-benchmarking/out/v1v2_8x16 \\
        --model-ref <vEcoli git SHA> --gen-lb 3 \\
        --s30-idx 5456 --s50-idx 5464 \\
        --out docs/report_cards/population_phenotype_basal/vs_vecoli/vecoli_reference.json

Layout: reference modes live as subdirs of the card they grade
(``<card>/vs_vecoli/``). "Equivalence" is a reference MODE, not a card.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from v2ecoli.library.card_vectors import extract_vectors

# ttest axes (per-cell distributions) -> the per-cell record field this computes.
_TTEST_AXES = {
    "physiology.doubling_time": "division_time",
    "physiology.cell_mass": "cell_mass_mean",
    "physiology.cell_volume": "volume_mean",
    "physiology.oric": "oric_mean",
    "physiology.replication_initiation": "replication_initiation_time",
    "physiology.replication_completion": "replication_completion_time",
    "composition.protein_fraction": "protein_fraction_mean",
    "composition.rna_fraction": "rna_fraction_mean",
    "composition.dna_fraction": "dna_fraction_mean",
    "ribosomes.total": "ribosome_total_mean",
    "ribosomes.active_fraction": "ribosome_active_fraction_mean",
    "ribosomes.elongation_rate": "ribosome_elongation_mean",
    "ribosomes.production": "ribosome_production_mean",
}
_OMICS_AXES = ["omics.transcriptome", "omics.proteome"]
_FLUX_SCATTER = "fluxes.exchange"

_DEFAULT_TEMPLATE = "tests/fixtures/population_phenotype_basal_reference.json"


def _replication_events(times, oric, nforks):
    init = next((times[i] for i in range(1, len(oric)) if oric[i] > oric[i - 1]), None)
    completion = next((times[i] for i in range(1, len(nforks))
                       if nforks[i] == 0 and nforks[i - 1] > 0), None)
    return init, completion


def _read_vecoli_cells(sweep_dir: str, s30_idx=None, s50_idx=None) -> list[dict]:
    """Cross-implementation reader for a vEcoli ("v1") parquet sweep — per-cell
    records (Physiology + Composition + Ribosomes) in the card's field shape.

    time normalized to since-birth (vEcoli `time` is cumulative); ribosome
    subunit counts sliced from the positional `bulk` list by integer index
    (vEcoli has no bulk__id/bulk__count). Cell-level aggregation throughout:
    time-mean WITHIN a cell, one value per cell."""
    import duckdb

    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        return []
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    avail = {d[0] for d in duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet({flist}, hive_partitioning=true)").fetchall()}
    tcol = "global_time" if "global_time" in avail else "time"
    have_ribo = (s30_idx is not None and s50_idx is not None
                 and "bulk" in avail
                 and "listeners__growth_limits__active_ribosome_allocated" in avail)
    # duckdb list_extract is 1-indexed; the indices are 0-indexed (Python).
    s30 = f"list_extract(bulk, {int(s30_idx) + 1})" if have_ribo else "NULL"
    s50 = f"list_extract(bulk, {int(s50_idx) + 1})" if have_ribo else "NULL"
    ribo = (", listeners__growth_limits__active_ribosome_allocated, "
            "listeners__ribosome_data__effective_elongation_rate, "
            "listeners__ribosome_data__total_rRNA_initiated, "
            f"{s30}, {s50}") if have_ribo else ", NULL, NULL, NULL, NULL, NULL"
    sel = (f"variant, lineage_seed, generation, agent_id, {tcol}, "
           "listeners__mass__dry_mass, listeners__mass__protein_mass, "
           "listeners__mass__rna_mass, listeners__mass__dna_mass, "
           "listeners__mass__cell_mass, listeners__mass__volume, "
           "listeners__replication_data__number_of_oric, "
           "len(listeners__replication_data__fork_coordinates)" + ribo)
    rows = duckdb.sql(
        f"SELECT {sel} FROM read_parquet({flist}, hive_partitioning=true) "
        f"ORDER BY variant, lineage_seed, generation, agent_id, {tcol}").fetchall()

    by_cell: dict[tuple, list] = {}
    for r in rows:
        by_cell.setdefault((int(r[0]), int(r[1]), int(r[2]), str(r[3])), []).append(r[4:])

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    records = []
    for (v, ls, g, a), rs in by_cell.items():
        t0 = rs[0][0]
        times = [x[0] - t0 for x in rs]
        orics = [x[7] for x in rs]; nforks = [x[8] for x in rs]
        d = [x[1] for x in rs]
        prot_f = [x[2] / x[1] for x in rs if x[1] > 0]
        rna_f = [x[3] / x[1] for x in rs if x[1] > 0]
        dna_f = [x[4] / x[1] for x in rs if x[1] > 0]
        repl_init, repl_complete = _replication_events(times, orics, nforks)
        rec = {
            "variant": v, "lineage_seed": ls, "generation": g, "agent_id": a,
            "division_time": float(rs[-1][0] - rs[0][0]),
            "cell_mass_mean": mean([x[5] for x in rs]),
            "volume_mean": mean([x[6] for x in rs]),
            "oric_mean": mean(orics),
            "replication_initiation_time": repl_init,
            "replication_completion_time": repl_complete,
            "protein_fraction_mean": mean(prot_f),
            "rna_fraction_mean": mean(rna_f),
            "dna_fraction_mean": mean(dna_f),
        }
        # Ribosomes (when bulk indices given): total = active + min(free 30S,50S);
        # active fraction = active/total; elongation + production from listeners.
        # x[9]=active, x[10]=elong, x[11]=rrna_init, x[12]=s30, x[13]=s50
        if len(rs[0]) > 9 and rs[0][9] is not None:
            totals, fracs, elongs, prods = [], [], [], []
            for x in rs:
                active = float(x[9] or 0.0)
                s30v = float(x[12] or 0.0); s50v = float(x[13] or 0.0)
                tot = active + min(s30v, s50v)
                totals.append(tot)
                if tot > 0:
                    fracs.append(active / tot)
                if (x[10] or 0.0) > 0:
                    elongs.append(float(x[10]))
                prods.append(float(x[11] or 0.0))
            rec["ribosome_total_mean"] = mean(totals)
            rec["ribosome_active_fraction_mean"] = mean(fracs)
            rec["ribosome_elongation_mean"] = mean(elongs)
            rec["ribosome_production_mean"] = mean(prods)
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", required=True, help="vEcoli parquet sweep dir")
    ap.add_argument("--template", default=_DEFAULT_TEMPLATE,
                    help="self-pin reference used as presentation/criterion template")
    ap.add_argument("--model-ref", required=True, help="vEcoli git commit (provenance)")
    ap.add_argument("--ensemble", default="", help="ensemble description (provenance)")
    ap.add_argument("--gen-lb", type=int, default=0,
                    help="generation_lower_bound: drop gens < this (burn-in)")
    ap.add_argument("--s30-idx", type=int, default=None,
                    help="0-indexed position of CPLX0-3953[c] (30S) in v1 bulk (sim_data)")
    ap.add_argument("--s50-idx", type=int, default=None,
                    help="0-indexed position of CPLX0-3962[c] (50S) in v1 bulk (sim_data)")
    ap.add_argument("--out", required=True, help="output reference json path")
    args = ap.parse_args()

    records = _read_vecoli_cells(args.sweep_dir, args.s30_idx, args.s50_idx)
    if args.gen_lb:
        records = [r for r in records if r["generation"] >= args.gen_lb]
    if not records:
        raise SystemExit(f"no cell records under {args.sweep_dir} (after gen-lb)")

    template = json.load(open(args.template))
    taxes = template["axes"]
    axes = {}
    print(f"v1 ensemble: {len(records)} cells")

    # --- ttest axes (Physiology + Composition + Ribosomes) ---
    for path, field in _TTEST_AXES.items():
        if path not in taxes:
            continue
        bycell = [[r["lineage_seed"], r["generation"], r[field]]
                  for r in records if r.get(field) is not None]
        vals = [v for _, _, v in bycell]
        if not vals:
            continue
        ax = dict(taxes[path]); ax["criterion"] = dict(ax["criterion"])
        ax["criterion"]["ref_values"] = vals
        ax["criterion"]["ref_by_cell"] = bycell
        axes[path] = ax
        print(f"  {path:36} n={len(vals):3}  mean={sum(vals) / len(vals):.4g}")

    # --- vector axes (Gene expression r2 + Exchange fluxes) ---
    vec = extract_vectors(args.sweep_dir, generation_lower_bound=args.gen_lb)
    omics = vec.get("omics") or {}
    for path in _OMICS_AXES:
        name = path.split(".")[1]
        node = omics.get(name) or {}
        if path not in taxes or not node.get("vector"):
            continue
        ax = dict(taxes[path]); ax["criterion"] = dict(ax["criterion"])
        ax["criterion"]["ref_vector"] = [round(x, 6) for x in node["vector"]]
        axes[path] = ax
        print(f"  {path:36} vector len={len(node['vector'])}")

    exch = (vec.get("fluxes") or {}).get("exchange") or {}
    per_cell = exch.get("per_cell") or []
    if _FLUX_SCATTER in taxes and exch.get("vector"):
        import numpy as np
        ref_std = (np.asarray(per_cell, float).std(axis=0).tolist()
                   if per_cell else [0.0] * len(exch["vector"]))
        ax = dict(taxes[_FLUX_SCATTER]); ax["criterion"] = dict(ax["criterion"])
        ax["criterion"]["ref_vector"] = [round(x, 8) for x in exch["vector"]]
        ax["criterion"]["ref_std"] = [round(float(x), 8) for x in ref_std]
        # flux_ids: v1 == v2 exactly (verified), so the template's order applies.
        axes[_FLUX_SCATTER] = ax
        print(f"  {_FLUX_SCATTER:36} vector len={len(exch['vector'])} n_cells={len(per_cell)}")
        # flux KPI ttest axes (slice per_cell by the template's flux_id index)
        flux_ids = ax["criterion"].get("flux_ids") or []
        idx = {m: i for i, m in enumerate(flux_ids)}
        for path, spec in taxes.items():
            crit = spec.get("criterion", {})
            fid = crit.get("flux_id")
            if not fid or path in axes or fid not in idx or not per_cell:
                continue
            j = idx[fid]
            kx = dict(spec); kx["criterion"] = dict(crit)
            kx["criterion"]["ref_values"] = [round(row[j], 8) for row in per_cell]
            axes[path] = kx
            print(f"  {path:36} KPI {fid} (idx {j})")

    out = {
        "$schema_note": template.get("$schema_note", ""),
        "title": "Basal-condition population phenotype — v1↔v2 equivalence",
        "status": "populated",
        "stimulus": {
            "reference_model": "vEcoli (v1)",
            "measured_model": "v2ecoli (v2)",
            "blessed_model_ref": args.model_ref,
            "ensemble": args.ensemble,
            "sweep_dir": args.sweep_dir,
            "generation_lower_bound": args.gen_lb,
        },
        "findings": [
            "EQUIVALENCE reference: v2ecoli graded against a vEcoli v1 ensemble "
            "(not a self-pin). Welch t-test of v2 cell-level values vs v1 ref_values; "
            "r2 for omics; flux_scatter for the exchange fingerprint.",
            "v1<->v2 share cistron/monomer/flux ordering exactly — vectors align "
            "positionally (no ID remapping). Ribosome 30S/50S sliced from v1's "
            "positional bulk by index (s30/s50 from v1 sim_data).",
            "v1 values read by scripts/pin_vecoli_equivalence_reference.py "
            "(self-contained cross-impl reader; shared analysis_runner is untouched).",
            "tolerance bands inherited from the self-pin template; revisit per-axis "
            "equivalence margins (delta) — and consider TOST for the formal claim.",
        ],
        "axes": axes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nwrote {args.out} ({len(axes)} axes)")


if __name__ == "__main__":
    main()
