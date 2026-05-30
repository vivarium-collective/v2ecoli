"""Post-sweep analysis runner.

Reads a finished sweep's emitted parquet (per-cell timeseries) + summary.json
(division metadata written by run_workflow), builds per-cell records, groups
them per scale, runs the AnalysisSteps named in analysis_options, and writes
analysis.json. Also runnable standalone:

    v2ecoli-analyze <sweep_dir> [--config cfg.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from typing import Any


def group_for_scale(scale: str, records: list[dict]) -> dict[tuple, list[dict]]:
    """Group per-cell records by the key the scale aggregates over."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        v, s = int(r["variant"]), int(r["lineage_seed"])
        g, a = int(r["generation"]), str(r["agent_id"])
        if scale == "single":
            key = (v, s, g, a)
        elif scale == "multidaughter":
            parent = a[:-1] if len(a) > 1 else a   # sisters share the parent id
            key = (v, s, g, parent)
        elif scale == "multigeneration":
            key = (v, s)
        elif scale == "multiseed":
            key = (v,)
        elif scale == "multivariant":
            key = ()
        else:
            continue
        groups.setdefault(key, []).append(r)
    return groups


_MASS_COLS = ("listeners__mass__dry_mass", "listeners__mass__protein_mass",
              "listeners__mass__rRna_mass", "listeners__mass__dna_mass")


def build_cell_records(sweep_dir: str) -> dict[tuple, dict]:
    """Build per-cell summary records from the sweep's parquet + summary.json."""
    import duckdb

    div_by_cell: dict[tuple, dict] = {}
    spath = os.path.join(sweep_dir, "summary.json")
    if os.path.isfile(spath):
        with open(spath) as f:
            summary = json.load(f)
        for bkey, bs in summary.items():
            m = re.search(r"variant=(\d+)/seed=(\d+)", bkey)
            if not m:
                continue
            v, s = int(m.group(1)), int(m.group(2))
            for gen in bs.get("generations", []):
                ck = (v, s, int(gen["generation"]), str(gen["agent_id"]))
                div_by_cell[ck] = {"divided": bool(gen.get("divided", False)),
                                   "division_time": float(gen.get("duration", 0.0))}

    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        return {}
    flist = "[" + ",".join("'" + f + "'" for f in files) + "]"
    sel = ("variant, lineage_seed, generation, agent_id, global_time, "
           + ", ".join(_MASS_COLS))
    rows = duckdb.sql(
        f"SELECT {sel} FROM read_parquet({flist}, hive_partitioning=true) "
        f"ORDER BY variant, lineage_seed, generation, agent_id, global_time"
    ).fetchall()

    by_cell: dict[tuple, list] = {}
    for row in rows:
        v, ls, g, a, t, dry, prot, rrna, dna = row
        ck = (int(v), int(ls), int(g), str(a))
        by_cell.setdefault(ck, []).append(
            (float(t), float(dry), float(prot), float(rrna), float(dna)))

    records: dict[tuple, dict] = {}
    for ck, rs in by_cell.items():
        fr = {"protein": [], "rRna": [], "dna": []}
        ts = []
        for (t, dry, prot, rrna, dna) in rs:
            ts.append({"listeners": {"mass": {"dry_mass": dry, "protein_mass": prot,
                                              "rRna_mass": rrna, "dna_mass": dna}}})
            if dry > 0:
                fr["protein"].append(prot / dry)
                fr["rRna"].append(rrna / dry)
                fr["dna"].append(dna / dry)
        div = div_by_cell.get(ck, {})
        records[ck] = {
            "variant": ck[0], "lineage_seed": ck[1], "generation": ck[2], "agent_id": ck[3],
            "divided": div.get("divided"),
            "division_time": div.get("division_time", float(rs[-1][0])),
            "newborn_dry_mass": rs[0][1], "final_dry_mass": rs[-1][1],
            "protein_fraction_mean": (sum(fr["protein"]) / len(fr["protein"])) if fr["protein"] else 0.0,
            "rRna_fraction_mean": (sum(fr["rRna"]) / len(fr["rRna"])) if fr["rRna"] else 0.0,
            "dna_fraction_mean": (sum(fr["dna"]) / len(fr["dna"])) if fr["dna"] else 0.0,
            "timeseries": ts,
        }
    return records


def _group_key_str(scale: str, key: tuple) -> str:
    if scale == "single":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/agent={key[3]}"
    if scale == "multidaughter":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/parent={key[3]}"
    if scale == "multigeneration":
        return f"variant={key[0]}/seed={key[1]}"
    if scale == "multiseed":
        return f"variant={key[0]}"
    return "all"


def run_analyses(sweep_dir: str, analysis_options: dict) -> dict:
    """Run the analyses named in ``analysis_options`` over the sweep's cells,
    write ``analysis.json``, and return the nested results."""
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, ANALYSIS_SCALES

    records = list(build_cell_records(sweep_dir).values())
    core = allocate_core()
    results: dict[str, dict] = {}
    for scale, analyses in (analysis_options or {}).items():
        if scale not in ANALYSIS_SCALES:
            warnings.warn(f"unknown analysis scale {scale!r}; skipping")
            continue
        groups = group_for_scale(scale, records)
        scale_out: dict[str, dict] = {}
        for name in (analyses or {}):
            step_cls = ANALYSIS_REGISTRY.get(name)
            if step_cls is None:
                warnings.warn(f"unknown analysis {name!r} (scale {scale}); skipping")
                continue
            if step_cls.scale != scale:
                warnings.warn(f"analysis {name!r} is scale {step_cls.scale}, "
                              f"not {scale}; skipping")
                continue
            step = step_cls({}, core=core)
            per_group: dict[str, Any] = {}
            for gkey, grp in groups.items():
                try:
                    rows = grp[0].get("timeseries") if scale == "single" else grp
                    per_group[_group_key_str(scale, gkey)] = step.analyze(rows or [])
                except Exception as e:
                    per_group[_group_key_str(scale, gkey)] = {
                        "error": f"{type(e).__name__}: {e}"}
            scale_out[name] = per_group
        results[scale] = scale_out

    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Run configured analyses over a sweep.")
    p.add_argument("sweep_dir", help="sweep output dir (parquet + summary.json)")
    p.add_argument("--config", default=None,
                   help="config JSON with analysis_options (with inherit_from)")
    args = p.parse_args()

    analysis_options: dict = {}
    if args.config:
        from v2ecoli.workflow.config import load_config_with_inheritance
        analysis_options = load_config_with_inheritance(args.config).get(
            "analysis_options") or {}
    if not analysis_options:
        print("no analysis_options found; nothing to run")
        return
    run_analyses(args.sweep_dir, analysis_options)
    print(f"Wrote {os.path.join(args.sweep_dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
