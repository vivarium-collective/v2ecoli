#!/usr/bin/env python
"""Functional flux diff: which metabolic reactions/pathways carry materially
different FBA flux between vEcoli and v2ecoli.

Operates on an existing compare_harness workdir (both engines already simulated).
Reads ``listeners__fba_results__base_reaction_fluxes`` (one signed flux per base
reaction per timestep, mmol/gDCW/h) from each engine's emitted parquet history,
time-averages each reaction's flux, matches reactions by ``base_reaction_id``,
and ranks the reactions whose mean flux differs materially between the models.

    .venv/bin/python scripts/compare_flux_pathways.py \
        --workdir out/compare_harness -o out/compare/flux_pathways.md

Base reactions lump isozymes / forward-reverse variants, so each row is a
biochemical conversion, and the reaction ID (an EcoCyc frame id) names the
pathway step. No general pathway taxonomy ships in the reconstruction, so we
annotate by amino-acid pathway (the one flat-file grouping available) plus a
coarse EC-class bucket, and otherwise let the reaction id speak for the step.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import duckdb
import numpy as np

FLUX_COL = "listeners__fba_results__base_reaction_fluxes"

# Materiality gates: ignore numerically negligible reactions (abs floor) and
# require a meaningful relative gap so float noise / tiny fluxes don't dominate.
ABS_FLOOR = 1e-3   # mmol/gDCW/h — below this a reaction is ~inactive in both
REL_GATE = 0.25    # >25% relative difference to count as "materially different"


def _mean_flux(hist_glob: str, n_expected: int) -> np.ndarray:
    """Time-average each reaction's signed flux across all emitted timesteps."""
    files = glob.glob(hist_glob, recursive=True)
    if not files:
        raise FileNotFoundError(f"no history parquet: {hist_glob}")
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    # UNNEST WITH ORDINALITY → (reaction_index, flux) long form, averaged in SQL
    # (avoids pulling every 2820-wide row into Python).
    res = con.execute(
        f"""
        SELECT idx, avg(f) AS mean_flux
        FROM (
            SELECT UNNEST({FLUX_COL}) AS f,
                   generate_subscripts({FLUX_COL}, 1) AS idx
            FROM read_parquet(?, union_by_name=true)
            WHERE {FLUX_COL} IS NOT NULL
        )
        GROUP BY idx ORDER BY idx
        """,
        [files],
    ).fetchnumpy()
    means = np.full(n_expected, np.nan)
    # generate_subscripts is 1-based
    means[res["idx"].astype(int) - 1] = res["mean_flux"]
    return means


def _load_vecoli_ids(parca_dir: Path) -> list[str]:
    with open(parca_dir / "kb" / "simData.cPickle", "rb") as f:
        sd = pickle.load(f)
    return list(sd.process.metabolism.base_reaction_ids)


def _load_v2_ids(parca_dir: Path) -> list[str]:
    with open(parca_dir / "checkpoint_step_9.pkl", "rb") as f:
        sd = pickle.load(f)
    return list(sd["metabolism"].base_reaction_ids)


def _aa_pathway_map() -> dict[str, str]:
    """reaction_id -> amino-acid pathway label, from the one flat-file grouping
    the reconstruction ships. Best-effort; absent file -> empty map."""
    for cand in [
        Path("/Users/eranagmon/code/vEcoli/reconstruction/ecoli/flat/amino_acid_pathways.tsv"),
    ]:
        if cand.exists():
            out = {}
            import csv
            with open(cand) as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    # columns vary; map any reaction-id-ish field to the aa label
                    rid = row.get("reaction id") or row.get("reaction_id") or row.get("rxn")
                    aa = row.get("amino acid") or row.get("amino_acid") or row.get("aa")
                    if rid and aa:
                        out[rid.strip('"')] = aa.strip('"')
            return out
    return {}


def _ec_bucket(rid: str) -> str:
    """Coarse pathway bucket from the EC class prefix of an EC-style id."""
    head = rid.split("-")[0]
    parts = head.split(".")
    if parts and parts[0].isdigit():
        return {
            "1": "oxidoreductase (redox)",
            "2": "transferase",
            "3": "hydrolase",
            "4": "lyase",
            "5": "isomerase",
            "6": "ligase",
            "7": "translocase/transport",
        }.get(parts[0], "other (EC)")
    return "named reaction"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", default="out/compare_harness")
    p.add_argument("-o", "--out", default="out/compare/flux_pathways.md")
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args(argv)
    work = Path(args.workdir)

    v_ids = _load_vecoli_ids(work / "vecoli_parca")
    v2_ids = _load_v2_ids(work / "v2_parca")
    v_flux = _mean_flux(
        str(work / "vecoli_sim" / "**" / "history" / "**" / "*.pq"), len(v_ids))
    v2_flux = _mean_flux(
        str(work / "v2_sim" / "parquet" / "**" / "history" / "**" / "*.pq"), len(v2_ids))

    v_map = {r: f for r, f in zip(v_ids, v_flux)}
    v2_map = {r: f for r, f in zip(v2_ids, v2_flux)}
    only_v = sorted(set(v_ids) - set(v2_ids))
    only_v2 = sorted(set(v2_ids) - set(v_ids))
    shared = [r for r in v_ids if r in v2_map]  # vEcoli order

    aa = _aa_pathway_map()
    rows = []
    for r in shared:
        a, b = float(v_map[r]), float(v2_map[r])
        if np.isnan(a):
            a = 0.0
        if np.isnan(b):
            b = 0.0
        absd = abs(a - b)
        rel = absd / max(abs(a), abs(b), 1e-12)
        # active in at least one engine and a real relative gap
        material = absd >= ABS_FLOOR and rel >= REL_GATE
        sign_flip = (a > ABS_FLOOR and b < -ABS_FLOOR) or (a < -ABS_FLOOR and b > ABS_FLOOR)
        rows.append({
            "rid": r, "v": a, "v2": b, "absd": absd, "rel": rel,
            "material": material, "flip": sign_flip,
            "pathway": aa.get(r, _ec_bucket(r)),
        })

    material_rows = sorted([x for x in rows if x["material"]],
                           key=lambda x: x["absd"], reverse=True)

    # pathway roll-up: total |Δflux| by annotated pathway bucket
    bucket = {}
    for x in material_rows:
        b = bucket.setdefault(x["pathway"], {"n": 0, "sum_absd": 0.0})
        b["n"] += 1
        b["sum_absd"] += x["absd"]
    pathway_rank = sorted(bucket.items(), key=lambda kv: kv[1]["sum_absd"], reverse=True)

    L = []
    L.append("# Functional flux diff — vEcoli vs v2ecoli\n")
    L.append(f"Source: `{work}` (2-generation sim, both engines, full-mode ParCa).\n")
    L.append("Each row is a **base reaction** (isozyme / fwd-rev variants lumped); "
             "flux is the per-reaction signed mean over all emitted timesteps "
             "(mmol·gDCW⁻¹·h⁻¹). Materiality: |Δ| ≥ "
             f"{ABS_FLOOR:g} and relative gap ≥ {REL_GATE:.0%}.\n")

    L.append("## Network structure\n")
    L.append(f"- base reactions: vEcoli **{len(v_ids)}**, v2ecoli **{len(v2_ids)}**, "
             f"shared **{len(shared)}**")
    L.append(f"- reactions only in vEcoli ({len(only_v)}): "
             + (", ".join(f"`{r}`" for r in only_v) if only_v else "—"))
    L.append(f"- reactions only in v2ecoli ({len(only_v2)}): "
             + (", ".join(f"`{r}`" for r in only_v2) if only_v2 else "—"))
    L.append(f"\n**{len(material_rows)} of {len(shared)} shared reactions carry "
             f"materially different flux.**\n")

    L.append("## Pathways with the largest flux divergence (Σ|Δflux| by bucket)\n")
    L.append("| pathway / EC bucket | # reactions | Σ\\|Δflux\\| |")
    L.append("|---|---:|---:|")
    for name, agg in pathway_rank[:20]:
        L.append(f"| {name} | {agg['n']} | {agg['sum_absd']:.3g} |")

    flips = [x for x in material_rows if x["flip"]]
    L.append(f"\n## Sign-flipped reactions (direction differs: {len(flips)})\n")
    if flips:
        L.append("| reaction | vEcoli | v2ecoli |")
        L.append("|---|---:|---:|")
        for x in flips[:30]:
            L.append(f"| `{x['rid']}` | {x['v']:.3g} | {x['v2']:.3g} |")
    else:
        L.append("_none — no reaction reverses direction between the models._")

    L.append(f"\n## Top {args.top} reactions by |Δflux|\n")
    L.append("| reaction | pathway/EC | vEcoli | v2ecoli | Δ | rel |")
    L.append("|---|---|---:|---:|---:|---:|")
    for x in material_rows[:args.top]:
        L.append(f"| `{x['rid']}` | {x['pathway']} | {x['v']:.3g} | {x['v2']:.3g} "
                 f"| {x['absd']:.3g} | {x['rel']:.0%} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"\n{len(material_rows)}/{len(shared)} shared reactions materially differ; "
          f"{len(only_v)} vEcoli-only, {len(only_v2)} v2ecoli-only, {len(flips)} sign-flipped")
    print("\nTop 15 by |Δflux|:")
    for x in material_rows[:15]:
        print(f"  {x['rid']:<42} v={x['v']:+.3g}  v2={x['v2']:+.3g}  "
              f"Δ={x['absd']:.3g}  ({x['rel']:.0%})  [{x['pathway']}]")


if __name__ == "__main__":
    main()
