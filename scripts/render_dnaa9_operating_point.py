#!/usr/bin/env python
"""Figures for the dnaa-9 initiation operating point (Rashmi 2026-07-01).

Operating point: RIDA_RATE_MULTIPLIER=0.05 + DNAA_INIT_LOW_THRESHOLD=8 +
DNAA_ASYNC_INITIATION=1 + DNAA_INIT_ECLIPSE_MIN=40. Satisfies all four asks:
DnaA-ATP fraction in [0.2,0.5], fires at full oriC-low occupancy (8/8),
asynchronous per-origin initiation (oriC 2->3->4, not the synchronous 2->4),
seed-robust.

Two figures:
  A. dnaa9_operating_point  — 4-panel lineage trajectory of the operating point
     (seed 0): cell-mass sawtooth, oriC count (1<->2 + async episodes), oriC-low
     occupancy (fires at full 1.0), and the TOTAL DnaA-ATP fraction sitting in the
     [0.2,0.5] band. Generation boundaries marked.
  B. dnaa9_async_contrast   — the asynchrony fix: oriC trajectory with async OFF
     (synchronous 2->4 jump) vs async ON (decoupled 2->3->4), same over-init config.

  python scripts/render_dnaa9_operating_point.py \
      --op out/opE40_s0 --sync out/ridaoff_sync --async out/ridaoff_async \
      --study-dir workspace/studies/dnaa-9-operating-point
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pbg_plot_style import PALETTE, style_axes, mark_lineages

RD = "listeners__replication_data__"
DNAA_APO, DNAA_ATP, DNAA_ADP = "PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"
B_ATP = [RD + "chromosomal_high_bound_atp", RD + "oric_high_bound_atp",
         RD + "oriC_low_bound_atp", RD + "promoter_high_bound_atp"]
B_ADP = [RD + "chromosomal_high_bound_adp", RD + "oric_high_bound_adp",
         RD + "promoter_high_bound_adp"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "svg.fonttype": "none"})


def _lineage(out_dir):
    return [p for p in glob.glob(os.path.join(out_dir, "**", "history", "**", "*.pq"),
                                 recursive=True) if re.search(r"agent_id=0+/", p + "/")]


def _load(out_dir, downsample=5000):
    files = _lineage(out_dir)
    if not files:
        return None, None
    avail = pl.read_parquet_schema(files[0])
    have = [c for c in B_ATP + B_ADP if c in avail]
    ids = pl.scan_parquet(files[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {m: ids.index(m) for m in (DNAA_APO, DNAA_ATP, DNAA_ADP)}
    bc = pl.col("bulk__count")
    cols = ["generation", "global_time", "listeners__mass__cell_mass",
            RD + "number_of_oric", RD + "oriC_low_bound_atp", RD + "oriC_low_free"]
    have = [c for c in have if c not in cols]  # dedupe (oriC_low_bound_atp is in both)
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains(r"^0+$"))
          .select(cols + have + [bc.list.get(v).alias(k) for k, v in idx.items()])
          .sort(["generation", "global_time"]).collect())
    # drop <5 min stub generations
    dur = (df.group_by("generation").agg(
        ((pl.col("global_time").max() - pl.col("global_time").min()) / 60.0).alias("_d")))
    df = df.filter(pl.col("generation").is_in(
        dur.filter(pl.col("_d") >= 5.0)["generation"].to_list()))
    # total DnaA-ATP fraction (free + bound ATP forms) / all DnaA
    present = set(cols + have)
    batp = sum([pl.col(c) for c in B_ATP if c in present]) if any(c in present for c in B_ATP) else pl.lit(0)
    badp = sum([pl.col(c) for c in B_ADP if c in present]) if any(c in present for c in B_ADP) else pl.lit(0)
    free = pl.col(DNAA_APO) + pl.col(DNAA_ATP) + pl.col(DNAA_ADP)
    df = df.with_columns([
        ((pl.col(DNAA_ATP) + batp) / pl.max_horizontal(free + batp + badp, pl.lit(1)))
        .alias("atp_frac_total"),
        (pl.col(DNAA_ATP) + pl.col(DNAA_ADP) + pl.col(DNAA_APO) + batp + badp).alias("total_dnaa"),
        (pl.col(RD + "oriC_low_bound_atp")
         / pl.max_horizontal(pl.col(RD + "oriC_low_bound_atp") + pl.col(RD + "oriC_low_free"),
                             pl.lit(1e-9))).alias("oric_low_occ"),
    ])
    # cumulative minutes + gen boundaries
    offset, cum, bounds = 0.0, [], []
    for g in sorted(df["generation"].unique().to_list()):
        t = df.filter(pl.col("generation") == g)["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0); offset += float(t.max()); bounds.append(offset / 60.0)
    df = df.with_columns(pl.Series("t_min", cum))
    if df.height > downsample:
        df = df.gather_every(max(1, df.height // downsample))
    return df, bounds[:-1]


def _oric(out_dir):
    df, b = _load(out_dir)
    return (df["t_min"].to_numpy(), df[RD + "number_of_oric"].to_numpy(), b) if df is not None else (None, None, None)


def operating_point(op_dir, study_dir, seed_label):
    df, bounds = _load(op_dir)
    if df is None:
        raise SystemExit(f"no data in {op_dir}")
    x = df["t_min"].to_numpy()
    afr = float(np.mean(df.filter(pl.col("generation") >= 4)["atp_frac_total"].to_numpy())) if df.height else 0.0
    fig, ax = plt.subplots(4, 1, figsize=(8.8, 9.6), sharex=True)
    ax[0].plot(x, df["listeners__mass__cell_mass"].to_numpy(), color=PALETTE["axis"], lw=1.4)
    style_axes(ax[0], "", "cell mass (fg)"); mark_lineages(ax[0], bounds)
    ax[0].set_title("Cell mass — regular division sawtooth")
    ax[1].step(x, df[RD + "number_of_oric"].to_numpy(), color=PALETTE["purple"], lw=1.5, where="post")
    style_axes(ax[1], "", "oriC count"); mark_lineages(ax[1], bounds)
    ax[1].set_ylim(0.5, 4.5); ax[1].set_yticks([1, 2, 3, 4])
    ax[1].set_title("oriC count — 1↔2 with asynchronous 2→3 episodes (decoupled origins, not 2→4)")
    ax[2].plot(x, df["oric_low_occ"].to_numpy(), color=PALETTE["green"], lw=1.2)
    style_axes(ax[2], "", "oriC-low occupancy"); mark_lineages(ax[2], bounds); ax[2].set_ylim(0, 1.05)
    ax[2].axhline(1.0, color=PALETTE["muted"], lw=0.8, ls=":")
    ax[2].set_title("oriC-low occupancy — fires at FULL saturation (8/8 = 1.0, threshold 8)")
    ax[3].axhspan(0.2, 0.5, color=PALETTE["blue"], alpha=0.10, lw=0)
    ax[3].plot(x, df["atp_frac_total"].to_numpy(), color=PALETTE["blue"], lw=1.2)
    style_axes(ax[3], "lineage time (min)", "DnaA-ATP fraction"); mark_lineages(ax[3], bounds)
    ax[3].set_ylim(0, 1.0)
    ax[3].set_title(f"Total DnaA-ATP fraction — in the Boesen [0.2,0.5] band (steady mean {afr:.2f})")
    fig.suptitle("dnaa-9 — initiation operating point: RIDA×0.05 + threshold 8 + async + eclipse 40 min\n"
                 f"({seed_label}) DnaA-ATP fraction in band · fires at full occupancy · asynchronous "
                 f"once-per-cycle", fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = os.path.join(study_dir, "charts", "dnaa9_operating_point")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-9 — initiation operating point satisfies all four targets (DnaA-ATP band, full occupancy, asynchrony, cycle)",
          f"Operating point RIDA×0.05 + threshold 8 + async + eclipse 40 min, {seed_label}. TOP→BOTTOM: "
          f"regular division sawtooth; oriC 1↔2 with asynchronous 2→3 episodes (origins decoupled — no "
          f"synchronous 2→4 jump); oriC-low occupancy fires at full 1.0 (threshold 8); total DnaA-ATP "
          f"fraction in the Boesen [0.2,0.5] band (mean {afr:.2f}). Seed-robust across 0/1/2.",
          "All four of Rashmi's requested behaviors hold together at one operating point, seed-robust: "
          "the DnaA-ATP fraction is in band (via RIDA, the regulatory hydrolysis), initiation fires at "
          "full oriC-low occupancy (threshold 8), origins initiate asynchronously (per-origin, oriC "
          "2→3→4), and the cell keeps a regular once-per-cycle division with DnaA homeostasis.",
          [os.path.basename(op_dir)], "scripts/render_dnaa9_operating_point.py")
    print(f"  wrote {base}.png/.svg (steady ATP fraction {afr:.2f})")


def async_contrast(sync_dir, async_dir, study_dir):
    xs, os_, bs = _oric(sync_dir)
    xa, oa, ba = _oric(async_dir)
    if xs is None or xa is None:
        print("  (async contrast skipped — missing data)"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax[0].step(xs, os_, color=PALETTE["red"], lw=1.4, where="post")
    mark_lineages(ax[0], bs); style_axes(ax[0], "lineage time (min)", "oriC count")
    ax[0].set_ylim(0.5, 4.5); ax[0].set_yticks([1, 2, 3, 4])
    ax[0].set_title("SYNCHRONOUS (async off): 2→4 jump", fontsize=10.5)
    ax[1].step(xa, oa, color=PALETTE["green"], lw=1.4, where="post")
    mark_lineages(ax[1], ba); style_axes(ax[1], "lineage time (min)", "")
    ax[1].set_ylim(0.5, 4.5); ax[1].set_yticks([1, 2, 3, 4])
    ax[1].set_title("ASYNCHRONOUS (async on): 2→3→4", fontsize=10.5)
    fig.suptitle("dnaa-9 — oriC asynchrony: per-origin initiation decouples the two origins\n"
                 "(same over-initiating config, RIDA off + threshold 8; async makes re-initiation "
                 "step 2→3→4 instead of the biologically-wrong synchronous 2→4)", fontsize=10.5, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    base = os.path.join(study_dir, "charts", "dnaa9_async_contrast")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".svg"); plt.close(fig)
    _meta(base + ".png",
          "dnaa-9 — asynchronous per-origin initiation replaces the synchronous 2→4 oriC jump with 2→3→4",
          "Same over-initiating stress config (RIDA off, threshold 8). LEFT (async off): when "
          "re-initiation occurs both origins fire in the same tick — oriC jumps 2→4 (biologically "
          "incorrect). RIGHT (async on): each origin initiates independently when its own low-affinity "
          "sites saturate — oriC steps 2→3→4. The origins separate via per-origin stochastic filling "
          "and the RIDA feedback (the first to fire drops free DnaA-ATP, delaying the other).",
          "The flag-gated asynchronous-initiation mechanism (DNAA_ASYNC_INITIATION) decouples the two "
          "origins as Rashmi requested: replication initiation is now a per-origin event, so overlapping "
          "rounds appear as oriC 2→3→4 rather than the synchronous 2→4 jump.",
          [os.path.basename(sync_dir), os.path.basename(async_dir)],
          "scripts/render_dnaa9_operating_point.py")
    print(f"  wrote {base}.png/.svg")


def _meta(path, title, caption, interp, runs, script):
    json.dump({"title": title, "caption": caption, "interpretation": interp,
               "source_runs": runs, "script": script}, open(path + ".meta.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="out/opE40_s0")
    ap.add_argument("--sync", default="out/ridaoff_sync")
    ap.add_argument("--asyncdir", dest="asyncdir", default="out/ridaoff_async")
    ap.add_argument("--study-dir", default="workspace/studies/dnaa-9-operating-point")
    ap.add_argument("--seed-label", default="seed 0")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.study_dir, "charts"), exist_ok=True)
    operating_point(args.op, args.study_dir, args.seed_label)
    async_contrast(args.sync, args.asyncdir, args.study_dir)


if __name__ == "__main__":
    main()
