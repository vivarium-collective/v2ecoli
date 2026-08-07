"""Pin a v1<->v2 *equivalence* reference for a population-phenotype card.

Unlike ``pin_population_phenotype_basal_reference.py`` (which self-pins v2ecoli
against its own blessed ensemble = a DRIFT instrument), this bakes a **vEcoli
("v1") ensemble** as the reference so the v2ecoli card grades as an EQUIVALENCE
instrument: "is v2 still the same E. coli as v1, within tolerance?"

**Any nutrient condition** (``--condition``, default ``basal``). Nothing in the
measurement depends on the condition — the readers below name listener columns
that every condition emits — so a condition is a *stimulus* label plus an output
location, not a code path. That is the point: five conditions are five
invocations, not five scripts. The template is shared for the same reason (it
carries presentation, criterion bands and flux ordering, all condition-independent);
only ``--out`` and the provenance stamp move.

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
    integer index, resolved BY MOLECULE ID from the sweep's own
    ``parca/kb/simData.cPickle`` when present (``--s30-idx``/``--s50-idx``
    override it, and are rejected if they disagree) for the
    ribosome total / active-fraction axes.
Omics/flux ensemble-mean vectors are read with the shared ``extract_vectors``
(column names match), since v1 emits the same listener columns.

``--sweep-dir`` may be a local path or an ``s3://`` URI (see
``v2ecoli.library.sweep_io``), so a reference can be pinned against a sweep that
never lands on local disk.

Cell selection mirrors ``PopulationPhenotypeBasalCard.analyze`` per axis (see
``_AXIS_FILTER``) — including dropping cells that never divided from the
doubling-time axis, read from the sweep's ``daughter_states/``. Both sides of an
equivalence card must select cells the same way, or the verdict confounds a
reader difference with an engine difference.

Run (the subunit indices resolve from the sweep's sim_data; pass them only to
override, e.g. for a sweep shipped without ``parca/kb/``):
    python scripts/pin_vecoli_equivalence_reference.py \\
        --sweep-dir /Users/chris/projects/SMS/vecoli-benchmarking/out/v1v2_8x16 \\
        --condition basal --model-ref <vEcoli git SHA> --gen-lb 3

Layout: reference modes live as subdirs of the card they grade
(``<card>/vs_vecoli/``). "Equivalence" is a reference MODE, not a card. The card
is named for its stimulus, so ``--condition acetate`` defaults ``--out`` to
``docs/report_cards/population_phenotype_acetate/vs_vecoli/vecoli_reference.json``
— parallel to the existing basal card rather than nested inside it, which leaves
``population_phenotype_basal`` untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from v2ecoli.library.card_vectors import extract_vectors
from v2ecoli.library.sweep_io import is_s3_uri

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
# Per-axis cell filters, mirroring PopulationPhenotypeBasalCard.analyze
# (v2ecoli/workflow/analysis.py) EXACTLY. Both sides of an equivalence card must
# select cells the same way, or the verdict confounds a reader difference with an
# engine difference. v2's three filters:
#   _divided — confirmed divisions only; a non-divided cell's division_time is the
#              per-generation duration cap, not a doubling time.
#   _pos     — fractions/levels: skip zero/absent (this reader's `mean()` yields
#              0.0, not None, for a cell with no valid timepoints).
#   _any     — event times: already None when the event didn't occur.
_FILTER_DIVIDED, _FILTER_POS, _FILTER_ANY = "divided", "pos", "any"
_AXIS_FILTER = {
    "physiology.doubling_time": _FILTER_DIVIDED,
    "physiology.cell_mass": _FILTER_POS,
    "physiology.cell_volume": _FILTER_POS,
    "physiology.oric": _FILTER_POS,
    "physiology.replication_initiation": _FILTER_ANY,
    "physiology.replication_completion": _FILTER_ANY,
    "composition.protein_fraction": _FILTER_POS,
    "composition.rna_fraction": _FILTER_POS,
    "composition.dna_fraction": _FILTER_POS,
    "ribosomes.total": _FILTER_POS,
    "ribosomes.active_fraction": _FILTER_POS,
    "ribosomes.elongation_rate": _FILTER_POS,
    "ribosomes.production": _FILTER_POS,
}


def _keep_cell(kind: str, rec: dict, value) -> bool:
    """Whether one cell contributes to an axis — v2's per-axis filter, on v1 records."""
    if value is None:
        return False
    if kind == _FILTER_DIVIDED:
        # `divided is None` = the sweep carries no signal; fall back to v2's other
        # condition (a positive duration) rather than dropping the whole axis.
        if rec.get("divided") is None:
            return float(value) > 0
        return rec.get("divided") is True and float(value) > 0
    if kind == _FILTER_POS:
        return float(value) > 0
    return True


_OMICS_AXES = ["omics.transcriptome", "omics.proteome"]
_FLUX_SCATTER = "fluxes.exchange"

_DEFAULT_TEMPLATE = "tests/fixtures/population_phenotype_basal_reference.json"


def _default_out(condition: str) -> str:
    """The vs_vecoli reference path for the card named after ``condition``."""
    return os.path.join("docs", "report_cards",
                        f"population_phenotype_{condition}", "vs_vecoli",
                        "vecoli_reference.json")


def _replication_events(times, oric, nforks):
    init = next((times[i] for i in range(1, len(oric)) if oric[i] > oric[i - 1]), None)
    completion = next((times[i] for i in range(1, len(nforks))
                       if nforks[i] == 0 and nforks[i - 1] > 0), None)
    return init, completion


def _divided_by_cell(sweep_dir: str) -> dict[tuple, bool] | None:
    """``{(variant, seed, generation, agent_id): True}`` for cells that divided.

    v2ecoli reads ``divided`` from a sweep's ``summary.json``; a vEcoli sweep has
    no such file, but it writes ``daughter_states/variant=V/seed=S/generation=G/
    agent_id=A/daughter_state_*.json`` when — and only when — a cell divides. So
    the presence of that directory is the v1 divided-signal.

    Returns ``None`` (not ``{}``) when the signal is unavailable — a remote sweep
    or a run without the directory — so the caller can tell "did not divide"
    apart from "cannot know", and decline to filter rather than silently drop
    every cell.
    """
    if is_s3_uri(sweep_dir):
        return None
    root = Path(sweep_dir) / "daughter_states"
    if not root.is_dir():
        return None
    out: dict[tuple, bool] = {}
    for agent_dir in root.glob("variant=*/seed=*/generation=*/agent_id=*"):
        if not any(agent_dir.glob("daughter_state_*.json")):
            continue
        try:
            parts = {p.split("=", 1)[0]: p.split("=", 1)[1]
                     for p in agent_dir.relative_to(root).parts}
            out[(int(parts["variant"]), int(parts["seed"]),
                 int(parts["generation"]), str(parts["agent_id"]))] = True
        except (KeyError, ValueError, IndexError):
            continue
    return out or None


def _resolve_subunit_indices(sweep_dir: str, s30_idx, s50_idx):
    """Resolve the 30S/50S bulk indices BY MOLECULE ID from the sweep's own sim_data.

    v2ecoli slices these subunits by id (``list_position(bulk__id, 'CPLX0-3953[c]')``);
    vEcoli emits ``bulk`` positionally with no id column, so this side has to slice
    by integer index. A wrong index does not raise — ``list_extract`` returns NULL,
    which reads downstream as a zero count, making ``total = active + min(s30, s50)``
    collapse to ``active`` and the active fraction to 1.0. Wrong, quietly, on all
    four ribosome axes.

    A vEcoli sweep ships ``parca/kb/simData.cPickle``, which carries the bulk id
    order and ``molecule_ids.s30_full_complex``/``s50_full_complex``. When it is
    readable, resolve the indices from it and treat any user-supplied value that
    disagrees as a hard error rather than a preference.
    """
    pickle_path = Path(sweep_dir) / "parca" / "kb" / "simData.cPickle"
    if is_s3_uri(sweep_dir) or not pickle_path.is_file():
        if s30_idx is not None and s50_idx is not None:
            print(f"  [warn] no sim_data at {pickle_path} — using the supplied "
                  f"--s30-idx/--s50-idx UNVALIDATED. A wrong index yields NULL "
                  f"(read as a zero count), not an error.")
        return s30_idx, s50_idx
    import pickle as _pickle
    try:
        with open(pickle_path, "rb") as f:
            sd = _pickle.load(f)
        ids = list(sd.internal_state.bulk_molecules.bulk_data["id"])
        want30 = sd.molecule_ids.s30_full_complex
        want50 = sd.molecule_ids.s50_full_complex
        got30, got50 = ids.index(want30), ids.index(want50)
    except Exception as exc:  # noqa: BLE001 — unreadable sim_data must not be fatal
        if s30_idx is not None and s50_idx is not None:
            print(f"  [warn] could not resolve subunit ids from {pickle_path} "
                  f"({type(exc).__name__}: {exc}) — using the supplied indices "
                  f"UNVALIDATED.")
        return s30_idx, s50_idx
    for label, supplied, resolved, mol in (("--s30-idx", s30_idx, got30, want30),
                                           ("--s50-idx", s50_idx, got50, want50)):
        if supplied is not None and int(supplied) != resolved:
            raise SystemExit(
                f"{label}={supplied} does not name {mol} in this sweep's sim_data "
                f"(that id is at index {resolved}). Slicing {supplied} would measure "
                f"a different molecule on every ribosome axis. Drop the flag and let "
                f"it resolve by id, or pass {resolved}.")
    print(f"  [bulk] {want30} -> {got30}, {want50} -> {got50} (resolved by id)")
    return got30, got50


def _read_vecoli_cells(sweep_dir: str, s30_idx=None, s50_idx=None) -> list[dict]:
    """Cross-implementation reader for a vEcoli ("v1") parquet sweep — per-cell
    records (Physiology + Composition + Ribosomes) in the card's field shape.

    time normalized to since-birth (vEcoli `time` is cumulative); ribosome
    subunit counts sliced from the positional `bulk` list by integer index
    (vEcoli has no bulk__id/bulk__count). Cell-level aggregation throughout:
    time-mean WITHIN a cell, one value per cell."""
    from v2ecoli.library.sweep_io import connect_for, history_files

    files = history_files(sweep_dir)
    if not files:
        return []
    divided_by_cell = _divided_by_cell(sweep_dir)
    s30_idx, s50_idx = _resolve_subunit_indices(sweep_dir, s30_idx, s50_idx)
    con = connect_for(sweep_dir)
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    avail = {d[0] for d in con.sql(
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
    rows = con.sql(
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
            # None (not False) when the sweep carries no divided-signal, so the
            # axis filter can decline to filter instead of dropping every cell.
            "divided": (None if divided_by_cell is None
                        else divided_by_cell.get((v, ls, g, a), False)),
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
    ap.add_argument("--sweep-dir", required=True,
                    help="vEcoli parquet sweep dir (local path or s3:// URI)")
    ap.add_argument("--condition", default="basal",
                    help="nutrient condition this ensemble was run under; sets the "
                         "card title, the provenance stamp and the default --out")
    ap.add_argument("--template", default=_DEFAULT_TEMPLATE,
                    help="self-pin reference used as presentation/criterion template "
                         "(condition-independent: presentation, bands, flux ordering)")
    ap.add_argument("--model-ref", required=True, help="vEcoli git commit (provenance)")
    ap.add_argument("--ensemble", default="", help="ensemble description (provenance)")
    ap.add_argument("--gen-lb", type=int, default=0,
                    help="generation_lower_bound: drop gens < this (burn-in)")
    ap.add_argument("--s30-idx", type=int, default=None,
                    help="override the 30S bulk index (default: resolve CPLX0-3953[c] by id\n                         from the sweep's parca/kb/simData.cPickle)")
    ap.add_argument("--s50-idx", type=int, default=None,
                    help="override the 50S bulk index (default: resolve CPLX0-3962[c] by id\n                         from the sweep's parca/kb/simData.cPickle)")
    ap.add_argument("--out", default=None,
                    help="output reference json path (default: the vs_vecoli "
                         "reference of the card named for --condition)")
    args = ap.parse_args()
    out_path = args.out or _default_out(args.condition)

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
        kind = _AXIS_FILTER.get(path, _FILTER_ANY)
        bycell = [[r["lineage_seed"], r["generation"], r[field]]
                  for r in records if _keep_cell(kind, r, r.get(field))]
        vals = [v for _, _, v in bycell]
        if not vals:
            continue
        if len(vals) < len(records):
            print(f"  [{path}] {len(records) - len(vals)} of {len(records)} cells "
                  f"dropped by the '{kind}' filter")
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
        # flux KPI ttest axes (slice per_cell by the template's flux_id index).
        # The index comes from the TEMPLATE's flux_ids but slices the MEASURED
        # vector, so the two must describe the same molecule set. A medium that
        # changes the external-exchange molecules changes that width, which would
        # otherwise slice silently-wrong values (or IndexError) — so check.
        flux_ids = ax["criterion"].get("flux_ids") or []
        width = len(exch["vector"])
        if flux_ids and len(flux_ids) != width:
            print(f"  [warn] template flux_ids ({len(flux_ids)}) != measured flux "
                  f"vector ({width}) — the exchange molecule set differs from the "
                  f"template's, so KPI axes indexed by name are not positionally "
                  f"safe. Re-pin flux_ids for this condition; skipping KPI axes.")
            flux_ids = []
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
        "title": (f"{args.condition} population phenotype — v1↔v2 equivalence"),
        "status": "populated",
        "stimulus": {
            "condition": args.condition,
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
            "v1 values read by scripts/pin_vecoli_equivalence_reference.py — a "
            "self-contained cross-impl reader for the two v1<->v2 emit-schema "
            "differences; it shares only sweep location/access "
            "(v2ecoli.library.sweep_io), not analysis logic.",
            "tolerance bands inherited from the self-pin template; revisit per-axis "
            "equivalence margins (delta) — and consider TOST for the formal claim.",
            ("The condition is a stimulus label: the same measurement runs unchanged "
             "on every nutrient condition, so a cross-condition read is a comparison "
             "of like-measured ensembles."),
        ],
        "axes": axes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nwrote {out_path} ({len(axes)} axes)")


if __name__ == "__main__":
    main()
