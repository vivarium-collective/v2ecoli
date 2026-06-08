"""dnaa-0 succinate baseline trajectory with a TRANSIENT vs STEADY-STATE split.

Expert-review improvement: the interpretation of this study hinges on the
distinction between the pre-steady-state TRANSIENT generations (gens 1-2, which
carry the cold-start initial transient) and the accepted periodic STEADY STATE
(gen 3 onward, where oriC must be periodic and cell mass periodic). This figure
shades those two regimes with different background colours and draws a labelled
divider at the gen-3 boundary, so the eye reads the acceptance window directly.

REAL RESULT (honest): in this seed-1 COLD-START run the oriC count is periodic
1<->2 in EVERY generation — it NEVER reaches 4. There is no 4-oriC overshoot in
the transient generations. The transient shading marks where the cold-start
initial condition is still relaxing, not a region of oriC overshoot; the panel
title and annotations state this explicitly rather than implying an overshoot
that does not occur.

Three panels on a shared lineage-time (min) axis:
  1. oriC count        (the acceptance read; real pattern 1<->2 throughout)
  2. cell mass (fg)    (periodic sawtooth from gen 3)
  3. total DnaA monomer (all 3 bulk forms; below-band un-tuned baseline)

Reads the workflow's FULL-history hive-partitioned parquet for the followed
(canonical all-zeros) daughter lineage; each generation's global_time resets,
so we offset each generation by the cumulative duration of prior generations.
The per-gen full-history emit bug is fixed, so each generation retains its
WHOLE cycle (~3000-5800 ticks/gen).

    python scripts/render_dnaa0_transient_split.py \
        --run out/dnaa0_fullhist/dnaa0_fullhist \
        --out studies/dnaa-0-succinate-baseline/charts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import polars as pl

from _run_provenance import run_id_from_run_dir

# bulk-store ids for the three DnaA nucleotide forms (apo / ATP / ADP).
DNAA_FORMS = ["PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"]

C = dict(oric="#9467bd", mass="#222222", dnaa="#2ca02c")
TRANSIENT_BG = "#fdecea"   # warm = pre-steady-state (allowed overshoot)
STEADY_BG = "#e8f3ec"      # cool = accepted steady-state window
STEADY_FIRST_GEN = 3       # gen 3 onward is the accepted steady state


def _load(run_dir: str):
    """Return (df with t_min, list of division-boundary times, gen-span list)."""
    files = glob.glob(os.path.join(run_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        files = glob.glob(os.path.join(run_dir, "**", "*.pq"), recursive=True)
    if not files:
        raise SystemExit(f"no history parquet under {run_dir}")
    ids = (pl.scan_parquet(files[0]).select("bulk__id").head(1)
           .collect()["bulk__id"][0].to_list())

    def _idx(mol: str) -> int:
        try:
            return ids.index(mol)
        except ValueError:
            raise SystemExit(f"bulk species {mol!r} absent from {run_dir}")

    fi = [_idx(m) for m in DNAA_FORMS]
    bc = pl.col("bulk__count")
    keep = ["generation", "global_time",
            "listeners__mass__cell_mass",
            "listeners__replication_data__number_of_oric"]
    df = (pl.scan_parquet(files, hive_partitioning=True)
          # follow ONLY the canonical all-zeros daughter (0, 00, 000, ...)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(keep + [(bc.list.get(fi[0]) + bc.list.get(fi[1])
                           + bc.list.get(fi[2])).alias("dnaa_total")])
          .sort(["generation", "global_time"]).collect())

    # drop sub-5-min daughter-stub generations (the final division spawns a
    # sliver generation that is not a real cell cycle).
    dur = (df.group_by("generation")
             .agg(((pl.col("global_time").max() - pl.col("global_time").min())
                   / 60.0).alias("_dur")))
    real = sorted(dur.filter(pl.col("_dur") >= 5.0)["generation"].to_list())
    df = df.filter(pl.col("generation").is_in(real))

    offset, cum, bounds, spans, gen_stats = 0.0, [], [], [], []
    for gen in real:
        sub = df.filter(pl.col("generation") == gen)
        t = sub["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0)
        start = offset / 60.0
        offset += float(t.max())
        spans.append((gen, start, offset / 60.0))
        bounds.append(offset / 60.0)
        gen_stats.append(dict(
            gen=int(gen),
            ticks=int(t.size),
            dur_min=round(float(t.max() - t.min()) / 60.0, 1),
            oric_set=sorted({int(v) for v in
                             sub["listeners__replication_data__number_of_oric"]
                             .to_list()}),
        ))
    df = df.with_columns(pl.Series("t_min", cum))
    n = df.height
    if n > 4000:
        df = df.gather_every(max(1, n // 4000))
    return df, bounds[:-1], spans, gen_stats


def render(run_dir: str, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df, bounds, spans, gen_stats = _load(run_dir)
    x = df["t_min"].to_numpy()
    oric = df["listeners__replication_data__number_of_oric"].to_numpy()
    oric_max = int(oric.max())
    # find the lineage-time where the accepted steady-state window begins
    steady_start = next((s for g, s, _ in spans if g >= STEADY_FIRST_GEN), None)
    x_end = float(x.max())

    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    # ----- regime background shading (transient vs accepted steady state) -----
    for a in ax:
        if steady_start is not None:
            a.axvspan(0, steady_start, color=TRANSIENT_BG, lw=0, zorder=0)
            a.axvspan(steady_start, x_end, color=STEADY_BG, lw=0, zorder=0)
        for b in bounds:
            a.axvline(b, color="0.0", lw=0.8, ls=":", alpha=0.3, zorder=1)
        a.grid(True, alpha=0.15, zorder=1)

    # bold labelled divider at the gen-3 boundary
    if steady_start is not None:
        for a in ax:
            a.axvline(steady_start, color="#b03030", lw=2.0, ls="--",
                      alpha=0.9, zorder=4)
        ax[0].text(steady_start, 1.02,
                   "steady-state acceptance window begins (gen 3)",
                   transform=ax[0].get_xaxis_transform(), ha="center",
                   va="bottom", fontsize=9, fontweight="bold", color="#b03030")

    ax[0].step(x, oric, color=C["oric"], lw=1.5, where="post", zorder=3)
    ax[0].set_ylabel("oriC\ncount")
    ax[0].set_ylim(0, max(4, oric_max) + 0.4)
    ax[0].set_yticks(range(0, max(4, oric_max) + 1))
    # acceptance annotation in each regime (honest: oriC is 1<->2 throughout)
    if steady_start is not None:
        ax[0].text(steady_start / 2, 0.06,
                   "TRANSIENT (cold-start)\noriC already 1↔2 (no overshoot)",
                   transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=8, color="#9a3b32")
        ax[0].text((steady_start + x_end) / 2, 0.06,
                   "STEADY STATE\nperiodic 1↔2 (oriC never 4)",
                   transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=8, color="#2f6b46")

    ax[1].plot(x, df["listeners__mass__cell_mass"], color=C["mass"], lw=1.3, zorder=3)
    ax[1].set_ylabel("cell mass\n(fg)")

    ax[2].plot(x, df["dnaa_total"], color=C["dnaa"], lw=1.3, zorder=3)
    ax[2].set_ylabel("total DnaA monomer\n(all forms, count)")
    ax[2].set_xlabel("lineage time (min, cumulative across generations)")

    # generation tags along the top panel
    for gen, start, end in spans:
        ax[0].text((start + end) / 2, 0.93, f"gen {gen}",
                   transform=ax[0].get_xaxis_transform(), ha="center", va="top",
                   fontsize=8, fontweight="bold", color="0.30")

    legend = [Patch(facecolor=TRANSIENT_BG, label="transient cold-start (gens 1–2)"),
              Patch(facecolor=STEADY_BG, label="accepted steady state (gen 3+)")]
    ax[0].legend(handles=legend, loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "dnaa-0 succinate baseline — transient vs accepted steady-state cell cycle\n"
        f"(seed 1 COLD-START, single daughters; full per-gen history; "
        f"oriC max = {oric_max} — REAL pattern is periodic 1↔2 every gen, NEVER 4)",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa0_transient_split")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "run_id": run_id_from_run_dir(run_dir),
        "source_run_id": "dnaa0_fullhist",
        "generation_id": None,
        "rendered_at": time.time(),
        "command": (f"python scripts/render_dnaa0_transient_split.py "
                    f"--run {run_dir} --out {out_dir}"),
        "note": (f"FULL per-generation history (per-gen-history emit fix). "
                 f"oriC max in this seed-1 COLD-START run = {oric_max}: the REAL "
                 "pattern is periodic 1<->2 in EVERY generation and NEVER 4 — "
                 "there is no 4-oriC overshoot. Background shading separates the "
                 "transient cold-start gens 1-2 from the accepted steady-state "
                 "window gen 3+; canonical all-zeros daughter lineage, sub-5-min "
                 "daughter-stub gens (e.g. gen 8) filtered."),
        "per_generation": gen_stats,
        "oric_max": oric_max,
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"wrote {base}.svg / .png  ({df.height} pts; oriC max {oric_max}; "
          f"gens {[g for g, _, _ in spans]})")
    for s in gen_stats:
        print(f"  gen{s['gen']}: ticks={s['ticks']} dur={s['dur_min']}min "
              f"oriC={s['oric_set']}")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa0_fullhist/dnaa0_fullhist")
    ap.add_argument("--out", default="studies/dnaa-0-succinate-baseline/charts")
    a = ap.parse_args()
    render(a.run, a.out)


if __name__ == "__main__":
    main()
