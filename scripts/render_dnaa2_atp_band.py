"""dnaa-2 DnaA-ATP fraction trace with the accepted target band SHADED.

Expert-review improvement: the pass criterion for the nucleotide-balance study
is that the DnaA-ATP fraction stays inside the accepted band [0.2, 0.5], so the
band must be drawn. This figure plots the generation-aligned DnaA-ATP fraction
trace (DnaA-ATP / total DnaA, from the three bulk forms) with:

  - the [0.2, 0.5] target band shaded,
  - per-generation average dots (the gen-mean fraction) overlaid,
  - generation dividers + labels,
  - in-band / out-of-band per-generation pass markers.

Reads the workflow's hive-partitioned history parquet for the followed
(all-zeros) daughter lineage of one seed.

    python scripts/render_dnaa2_atp_band.py \
        --run out/dnaa2_seed1_8gen --seed 1 \
        --out studies/dnaa-2-nucleotide-balance/charts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import polars as pl

# DnaA-ATP / DnaA-ADP / apo-DnaA bulk ids.
DATP, DADP, DAPO = "MONOMER0-160[c]", "MONOMER0-4565[c]", "PD03831[c]"
BAND_LO, BAND_HI = 0.2, 0.5
C_TRACE = "#1f77b4"
C_BAND = "#1f77b4"
C_IN = "#2ca02c"
C_OUT = "#d62728"


def _load(run_dir: str, seed: int):
    files = glob.glob(os.path.join(
        run_dir, "**", "history", "**", f"lineage_seed={seed}", "**", "*.pq"),
        recursive=True)
    if not files:
        files = glob.glob(os.path.join(run_dir, "**", "history", "**", "*.pq"),
                          recursive=True)
    if not files:
        raise SystemExit(f"no history parquet for seed {seed} under {run_dir}")
    ids = (pl.scan_parquet(files[0]).select("bulk__id").head(1)
           .collect()["bulk__id"][0].to_list())
    ai, di, oi = ids.index(DATP), ids.index(DADP), ids.index(DAPO)
    bc = pl.col("bulk__count")
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(["generation", "global_time",
                   bc.list.get(ai).alias("atp"),
                   bc.list.get(di).alias("adp"),
                   bc.list.get(oi).alias("apo")])
          .sort(["generation", "global_time"]).collect())
    dur = (df.group_by("generation")
             .agg(((pl.col("global_time").max() - pl.col("global_time").min())
                   / 60.0).alias("_d")))
    real = sorted(dur.filter(pl.col("_d") >= 5.0)["generation"].to_list())
    df = df.filter(pl.col("generation").is_in(real))
    df = df.with_columns(
        (pl.col("atp") / (pl.col("atp") + pl.col("adp") + pl.col("apo")))
        .alias("frac"))

    offset, cum, bounds, gen_pts = 0.0, [], [], []
    for gen in real:
        s = df.filter(pl.col("generation") == gen)
        t = s["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0)
        start = offset / 60.0
        offset += float(t.max())
        end = offset / 60.0
        gen_pts.append((gen, (start + end) / 2.0,
                        float(np.nanmean(s["frac"].to_numpy()))))
        bounds.append(end)
    df = df.with_columns(pl.Series("t_min", cum))
    n = df.height
    if n > 4000:
        df = df.gather_every(max(1, n // 4000))
    return df, bounds[:-1], gen_pts


def render(run_dir: str, seed: int, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, bounds, gen_pts = _load(run_dir, seed)
    x = df["t_min"].to_numpy()
    frac = df["frac"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # shaded accepted target band + edge lines
    ax.axhspan(BAND_LO, BAND_HI, color=C_BAND, alpha=0.12, lw=0, zorder=0,
               label=f"accepted band [{BAND_LO}, {BAND_HI}]")
    ax.axhline(BAND_LO, color=C_BAND, lw=0.9, ls="--", alpha=0.5, zorder=1)
    ax.axhline(BAND_HI, color=C_BAND, lw=0.9, ls="--", alpha=0.5, zorder=1)

    for b in bounds:
        ax.axvline(b, color="0.0", lw=0.8, ls=":", alpha=0.3, zorder=1)

    ax.plot(x, frac, color=C_TRACE, lw=1.1, alpha=0.85, zorder=2,
            label="DnaA-ATP fraction (per tick)")

    # per-generation average dots, coloured by in/out of band
    n_in = 0
    for gen, gx, gm in gen_pts:
        inband = BAND_LO <= gm <= BAND_HI
        n_in += int(inband)
        ax.plot(gx, gm, "o", ms=10, mec="white", mew=1.2,
                color=C_IN if inband else C_OUT, zorder=5)
        ax.annotate(f"{gm:.2f}", (gx, gm), textcoords="offset points",
                    xytext=(0, 12 if gm < 0.45 else -16), ha="center",
                    fontsize=7.5, fontweight="bold",
                    color=C_IN if inband else C_OUT)
        ax.text(gx, 0.02, f"gen {gen}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="0.30")

    ax.set_ylim(0, max(1.02, float(np.nanmax(frac)) + 0.05))
    ax.set_xlabel("lineage time (min, cumulative across generations)")
    ax.set_ylabel("DnaA-ATP fraction\n(DnaA-ATP / total DnaA)")
    ax.grid(True, alpha=0.15)

    # marker legend entries
    ax.plot([], [], "o", color=C_IN, mec="white", label="gen-mean in band")
    ax.plot([], [], "o", color=C_OUT, mec="white", label="gen-mean out of band")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=2)

    fig.suptitle(
        "dnaa-2 nucleotide balance — DnaA-ATP fraction vs accepted band [0.2, 0.5]\n"
        f"(seed {seed}, {len(gen_pts)} generations; {n_in}/{len(gen_pts)} "
        "gen-means in band; gen 1 = pre-steady-state transient)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa2_atp_fraction_band")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "source_run_id": f"dnaa2_seed{seed}_8gen",
        "generation_id": None,
        "rendered_at": time.time(),
        "command": (f"python scripts/render_dnaa2_atp_band.py --run {run_dir} "
                    f"--seed {seed} --out {out_dir}"),
        "note": (f"{n_in}/{len(gen_pts)} per-generation mean DnaA-ATP fractions "
                 f"inside accepted band [{BAND_LO}, {BAND_HI}]. Fraction derived "
                 "from the three bulk DnaA forms (ATP/ADP/apo)."),
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"wrote {base}.svg / .png  ({df.height} pts; gen-means "
          f"{[round(m,3) for _,_,m in gen_pts]}; {n_in}/{len(gen_pts)} in band)")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa2_seed1_8gen")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="studies/dnaa-2-nucleotide-balance/charts")
    a = ap.parse_args()
    render(a.run, a.seed, a.out)


if __name__ == "__main__":
    main()
