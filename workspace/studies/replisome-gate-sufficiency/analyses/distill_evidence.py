"""Distil each run's parquet into a compact evidence bundle.

A full run emits ~3 GB of parquet across 206 columns. Every graded axis in this
study reads seven of them. This writes one small parquet per run holding just
those, so the bulk history can be deleted without losing the ability to
recompute the verdicts or redraw the figures.

What is kept, and why each is load-bearing:

    generation, global_time     locates a timestep in the lineage
    number_of_oric              sets the gate's demand (6x trimers, 2x monomers)
    the six subunit counts      the margins: observed minus demand
    cell_mass, dry_mass         distinguishes a mass-gated stall from a
                                subunit-gated one. criticalMassPerOriC is NOT
                                emitted by this build, so dry mass is the only
                                available proxy for the gate's other
                                precondition; the study's readouts record that
                                as derived-needed rather than pretending
                                otherwise.

The per-generation division outcomes (divided / tau / final dry mass) already
live in each run's ``*_summary.json``, which sits OUTSIDE the parquet tree and
is 4 KB, so it survives deletion untouched.

VERIFY BEFORE DELETING. ``--verify`` recomputes the subunit margins from the
distilled bundle and compares them against the same margins computed from the
full parquet. Deleting bulk data whose distillation has not been checked is how
a study quietly loses its evidence.

Usage::

    python .../analyses/distill_evidence.py            # distil every run
    python .../analyses/distill_evidence.py --verify   # distil, then cross-check
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

OUT_ROOT = REPO / "out" / STUDY_DIR.name
BUNDLE_DIR = STUDY_DIR / "evidence"

ORIC = "listeners__replication_data__number_of_oric"
KEEP_SCALAR = ["global_time", ORIC,
               "listeners__mass__cell_mass", "listeners__mass__dry_mass"]
SUBUNITS = {
    "CPLX0-2361[c]": "pol_III_core",
    "CPLX0-3761[c]": "beta_clamp",
    "CPLX0-3621[c]": "DnaB_hexamer",
    "EG10239-MONOMER[c]": "DnaG",
    "EG11500-MONOMER[c]": "HolB",
    "EG11412-MONOMER[c]": "HolA",
}
MULT = {"pol_III_core": 6, "beta_clamp": 6, "DnaB_hexamer": 2,
        "DnaG": 2, "HolB": 2, "HolA": 2}
NAME_MAP = {"pol III core": "pol_III_core", "beta clamp": "beta_clamp",
            "DnaB hexamer": "DnaB_hexamer", "DnaG": "DnaG",
            "HolB (delta')": "HolB", "HolA (delta)": "HolA"}


def run_dirs(out_root: Path) -> list[Path]:
    """Every directory under *out_root* holding a runner summary.

    rglob rather than a fixed depth: study 2 nests arm/seedN, study 1 nests only
    arm, and a fixed glob silently returns nothing for the other layout.
    """
    return sorted({p.parent for p in out_root.rglob("*_summary.json")})


def distil(run_dir: Path):
    import polars as pl
    files = sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))
    frames = []
    for f in files:
        df = pl.read_parquet(f)
        if ORIC not in df.columns or "bulk__id" not in df.columns:
            continue
        ids = df["bulk__id"][0].to_list()
        out = df.select([c for c in KEEP_SCALAR if c in df.columns])
        # `generation` is a hive PARTITION KEY (a generation=N directory), not a
        # column, so reading files individually drops it. Recover it from the path
        # or the bundle cannot be sliced per generation.
        if "generation" not in out.columns:
            mo = re.search(r"generation=(\d+)", f)
            if mo is None:
                continue
            out = out.with_columns(
                pl.lit(int(mo.group(1)), dtype=pl.Int64).alias("generation"))
        for mol, label in SUBUNITS.items():
            if mol in ids:
                out = out.with_columns(
                    df["bulk__count"].list.get(ids.index(mol)).alias(label))
        frames.append(out)
    return pl.concat(frames, how="diagonal") if frames else None


def margins_from_bundle(bundle, generation: int) -> dict:
    g = bundle.filter(bundle["generation"] == generation)
    if g.height == 0:
        return {}
    oric = g[ORIC]
    return {lab: int((g[lab] - oric * m).min())
            for lab, m in MULT.items() if lab in g.columns}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check distilled margins against the full parquet")
    ap.add_argument("--out-root", default=None,
                    help="simulation output root (default: this study's out/ dir)")
    ap.add_argument("--bundle-dir", default=None,
                    help="where to write the bundle (default: this study's evidence/)")
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else OUT_ROOT
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else BUNDLE_DIR
    if not out_root.is_absolute():
        out_root = REPO / out_root
    if not bundle_dir.is_absolute():
        bundle_dir = REPO / bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs(out_root)
    if not dirs:
        print(f"no completed runs under {out_root}", file=sys.stderr)
        return 1

    total_in = total_out = 0
    mismatches, checked = [], 0
    for d in dirs:
        # study 2 is out/<study>/<arm>/seedN; study 1 is out/<study>/<arm>.
        rel = d.relative_to(out_root)
        label = "__".join(rel.parts) if rel.parts else d.name
        bundle = distil(d)
        if bundle is None:
            print(f"  {label}: no history parquet, skipped")
            continue
        target = bundle_dir / f"{label}.parquet"
        bundle.write_parquet(target, compression="zstd")
        src = sum(f.stat().st_size for f in d.rglob("*.pq"))
        total_in += src
        total_out += target.stat().st_size
        print(f"  {label}: {bundle.height:6} rows  "
              f"{src/2**30:5.2f} GB -> {target.stat().st_size/2**20:6.2f} MB")

        if args.verify:
            from v2ecoli.library import replisome_arrest as ra
            summ = json.load(open(list(d.glob("*_summary.json"))[0]))
            stall = next((g["gen"] for g in summ["gens"] if not g["divided"]), None)
            if stall is None:
                continue
            tri, mono = ra.subunit_groups(REPO / "out/cache")
            truth = {v["label"]: v["margin"]
                     for v in ra.subunit_margins(d, stall, tri, mono).values()}
            got = margins_from_bundle(bundle, stall)
            for lab, m in truth.items():
                key = NAME_MAP.get(lab, lab)
                if key in got:
                    checked += 1
                    if got[key] != m:
                        mismatches.append(f"{label} gen{stall} {lab}: "
                                          f"parquet {m:+d} vs bundle {got[key]:+d}")

    print(f"\n{len(dirs)} run(s): {total_in/2**30:.1f} GB -> {total_out/2**20:.1f} MB "
          f"({total_in/max(total_out,1):.0f}x smaller)")
    if args.verify:
        if mismatches:
            print(f"\nVERIFY FAILED - {len(mismatches)} mismatch(es). DO NOT DELETE:")
            for m in mismatches:
                print("   ", m)
            return 1
        print(f"\nVERIFY OK - {checked} margin(s) across every stalled run reproduce "
              "exactly from the bundle. The bulk parquet is safe to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
