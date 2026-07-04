"""dnaa-1 expression-tuning DECISION figure (FULL-history, 4 panels).

Connects the Mechanism-A intervention (raised dnaA transcription, V=1.7e-3) to
its molecular consequence and the preserved cell-cycle phenotype, with every
panel sharing the SAME cumulative lineage-time (min) axis and showing the FULL
within-cycle trajectory of each generation:

  (a) dnaA mRNA initiation events per minute    (per-60s bin of the dnaA cistron)
  (b) total DnaA monomer count                  (sum of the 3 bulk DnaA forms;
                                                 accepted band [300, 800] shaded)
  (c) DnaA concentration (nM)                   (total / (volume * N_A))
  (d) oriC count                                (cell-cycle phenotype)

DATA SOURCE
-----------
Reads the workflow's hive-partitioned history parquet for the followed
(canonical all-zeros agent_id) daughter lineage. The per-gen full-history emit
bug is fixed (commit 2cc0ebf), so each generation now retains its WHOLE cycle.
Each generation's global_time resets, so generations are offset by the
cumulative duration of prior generations.

The dnaA total is the sum of the three bulk DnaA nucleotide forms
(apo PD03831[c] + DnaA-ATP MONOMER0-160[c] + DnaA-ADP MONOMER0-4565[c]) — the
honest total across forms, in the [300,800]-band scale. (The monomer_counts
listener index reports a much larger, non-DnaA-scale value here and is NOT
used.)

    python scripts/render_dnaa1_decision.py \
        --run out/dnaa1_fullhist/dnaa1_fullhist \
        --out studies/dnaa-1-expression/charts
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

import numpy as np
import polars as pl

from _run_provenance import run_id_from_run_dir

# bulk-store ids for the three DnaA nucleotide forms (apo / ATP / ADP).
DNAA_FORMS = ["PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"]
DNAA_CISTRON_IDX = 227          # EG10235 (dnaA) in rna_init_event_per_cistron
AVOGADRO = 6.022e23
BAND_LO, BAND_HI = 300, 800     # accepted total-DnaA band (counts)
STUB_MIN = 5.0                  # generations shorter than this are daughter stubs

C = dict(init="#8c564b", dnaa="#222222", conc="#ff7f0e", oric="#9467bd")


def _load(run_dir: str):
    """Return (df with t_min, division-boundary times, gen-span list)."""
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
    init_col = "listeners__rnap_data__rna_init_event_per_cistron"
    keep = ["generation", "global_time",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__replication_data__number_of_oric"]
    df = (pl.scan_parquet(files, hive_partitioning=True)
          # follow ONLY the canonical all-zeros daughter (0, 00, 000, ...)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(keep + [
              (bc.list.get(fi[0]) + bc.list.get(fi[1])
               + bc.list.get(fi[2])).alias("dnaa_total"),
              pl.col(init_col).list.get(DNAA_CISTRON_IDX, null_on_oob=True)
                .fill_null(0).cast(pl.Int64).alias("dnaa_init"),
          ])
          .sort(["generation", "global_time"]).collect())

    # drop sub-STUB_MIN daughter-stub generations (1-tick birth stubs etc.)
    dur = (df.group_by("generation")
             .agg(((pl.col("global_time").max() - pl.col("global_time").min())
                   / 60.0).alias("_dur")))
    real = sorted(dur.filter(pl.col("_dur") >= STUB_MIN)["generation"].to_list())
    df = df.filter(pl.col("generation").is_in(real))

    # DnaA concentration (nM) = count / (volume[fL] * N_A * 1e-15) * 1e9
    df = df.with_columns(
        (pl.col("dnaa_total")
         / (pl.col("listeners__mass__volume") * AVOGADRO * 1e-15) * 1e9
         ).alias("conc_nM"))

    # cumulative lineage time + per-gen init RATE (events/min in 60-s bins)
    offset, cum, bounds, spans, gen_stats = 0.0, [], [], [], []
    rate_x, rate_y = [], []
    for gen in real:
        sub = df.filter(pl.col("generation") == gen)
        t = sub["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0)
        start = offset / 60.0
        gen_start_abs = offset
        offset += float(t.max())
        spans.append((gen, start, offset / 60.0))
        bounds.append(offset / 60.0)
        # init rate: bin the dnaA init events into 1-min windows
        ev = sub["dnaa_init"].to_numpy()
        if t.size > 1:
            tmin_g = (t - t.min()) / 60.0
            nbin = max(1, int(np.ceil(tmin_g.max())))
            edges = np.arange(nbin + 1)
            for b in range(nbin):
                m = (tmin_g >= edges[b]) & (tmin_g < edges[b + 1])
                if m.any():
                    span = max(edges[b + 1] - edges[b], 1e-9)
                    rate_y.append(float(ev[m].sum()) / span)
                    rate_x.append((gen_start_abs / 60.0) + (edges[b] + edges[b + 1]) / 2.0)
        gen_stats.append(dict(
            gen=int(gen),
            ticks=int(t.size),
            dur_min=round(float(t.max() - t.min()) / 60.0, 1),
            dnaa_mean=round(float(sub["dnaa_total"].mean()), 1),
            dnaa_min=int(sub["dnaa_total"].min()),
            dnaa_max=int(sub["dnaa_total"].max()),
            oric_set=sorted({int(v) for v in
                             sub["listeners__replication_data__number_of_oric"]
                             .to_list()}),
            init_per_min=round(float(ev.sum()) / max((t.max() - t.min()) / 60.0,
                                                     1e-9), 2),
        ))
    df = df.with_columns(pl.Series("t_min", cum))

    # decimate for plotting line panels
    n = df.height
    if n > 6000:
        df = df.gather_every(max(1, n // 6000))
    return df, bounds[:-1], spans, np.array(rate_x), np.array(rate_y), gen_stats


def render(run_dir: str, out_dir: str, v_label: str = "1.7e-3") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, bounds, spans, rate_x, rate_y, gen_stats = _load(run_dir)
    x = df["t_min"].to_numpy()
    dnaa = df["dnaa_total"].to_numpy()
    conc = df["conc_nM"].to_numpy()
    oric = df["listeners__replication_data__number_of_oric"].to_numpy()
    oric_max = int(oric.max())

    fig, ax = plt.subplots(4, 1, figsize=(11, 11.5), sharex=True)

    # generation boundaries on every panel
    for a in ax:
        for b in bounds:
            a.axvline(b, color="0.0", lw=0.8, ls=":", alpha=0.3, zorder=1)
        a.grid(True, alpha=0.15, zorder=1)

    # (a) dnaA mRNA initiation rate (events/min, per-minute bins)
    if rate_x.size:
        ax[0].plot(rate_x, rate_y, color=C["init"], lw=1.0, alpha=0.55, zorder=2)
        # smoothed overlay (rolling mean) for readability
        if rate_y.size >= 5:
            k = 5
            sm = np.convolve(rate_y, np.ones(k) / k, mode="same")
            ax[0].plot(rate_x, sm, color=C["init"], lw=2.0, zorder=3)
    ax[0].axhline(1.0, color="0.4", lw=1.0, ls="--", alpha=0.7,
                  label="biological target ≈1/min")
    ax[0].set_ylabel("dnaA mRNA init\n(events/min)")
    ax[0].set_ylim(bottom=0)
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    # (b) total DnaA monomer count (with accepted band)
    ax[1].axhspan(BAND_LO, BAND_HI, color="0.5", alpha=0.12, lw=0,
                  label=f"accepted band [{BAND_LO}, {BAND_HI}]")
    ax[1].plot(x, dnaa, color=C["dnaa"], lw=1.2, zorder=3)
    ax[1].set_ylabel("total DnaA\n(counts, all forms)")
    ax[1].legend(loc="upper left", fontsize=8, framealpha=0.9)

    # (c) DnaA concentration
    ax[2].plot(x, conc, color=C["conc"], lw=1.2, zorder=3)
    ax[2].set_ylabel("DnaA conc\n(nM)")

    # (d) oriC count
    ax[3].step(x, oric, color=C["oric"], lw=1.4, where="post", zorder=3)
    ax[3].set_ylabel("oriC\ncount")
    ax[3].set_ylim(0, max(4, oric_max) + 0.4)
    ax[3].set_yticks(range(0, max(4, oric_max) + 1))
    ax[3].set_xlabel("lineage time (min, cumulative across generations)")

    # generation tags along the top
    for gen, start, end in spans:
        ax[0].text((start + end) / 2, 0.93, f"gen {gen}",
                   transform=ax[0].get_xaxis_transform(), ha="center", va="top",
                   fontsize=8, fontweight="bold", color="0.30")

    fig.suptitle(
        f"dnaa-1 expression tuning — DECISION figure (Mechanism A, V={v_label}, seed 1)\n"
        "intervention → molecular consequence → preserved cell-cycle phenotype\n"
        "(full within-cycle trajectories; all panels on a shared lineage-time axis)",
        fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa1_decision")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "run_id": run_id_from_run_dir(run_dir),
        "source_run_id": os.path.basename(run_dir.rstrip("/")),
        "generation_id": None,
        "rendered_at": time.time(),
        "command": (f"python scripts/render_dnaa1_decision.py --run {run_dir} "
                    f"--out {out_dir} --v-label {v_label}"),
        "v_label": v_label,
        "note": ("FULL per-generation history (per-gen-history emit fix, commit "
                 "2cc0ebf). 4 panels on a shared cumulative lineage-time axis: "
                 "(a) dnaA mRNA init events/min (per-min bins, dnaA cistron 227); "
                 "(b) total DnaA = sum of bulk forms PD03831[c]+MONOMER0-160[c]+"
                 "MONOMER0-4565[c], band [300,800]; (c) DnaA conc nM; "
                 "(d) oriC count. Canonical all-zeros daughter lineage; "
                 "daughter-stub gens (<5 min) filtered."),
        "per_generation": gen_stats,
        "oric_max": oric_max,
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"wrote {base}.svg / .png  ({df.height} plotted pts; oriC max {oric_max})")
    for s in gen_stats:
        print(f"  gen{s['gen']}: ticks={s['ticks']} dur={s['dur_min']}min "
              f"DnaA mean={s['dnaa_mean']} [{s['dnaa_min']},{s['dnaa_max']}] "
              f"oriC={s['oric_set']} init={s['init_per_min']}/min")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa1_fullhist/dnaa1_fullhist")
    ap.add_argument("--out", default="studies/dnaa-1-expression/charts")
    ap.add_argument("--v-label", default="1.7e-3",
                    help="V (TU00259[c] synth prob) label for the figure title/meta")
    a = ap.parse_args()
    render(a.run, a.out, v_label=a.v_label)


if __name__ == "__main__":
    main()
