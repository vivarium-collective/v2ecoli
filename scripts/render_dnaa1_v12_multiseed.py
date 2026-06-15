"""dnaa-1 V=1.2e-3 MULTISEED figure (Rashmi's request, 2026-06-08).

Rashmi asked for a multiseed simulation at V=1.2e-3 to see whether DnaA levels
oscillate within the accepted band [300, 800] while MAINTAINING steady-state
cell-cycle dynamics (periodic mass-doubling division and oriC staying 1<->2
with no reinitiation to 4).

For each seed it computes, over the steady generations (default gens 4-7) on the
canonical all-zeros daughter lineage (daughter-stub gens < STUB_MIN dropped):

  - DnaA monomer pool envelope: steady-gen trough / cycle-mean / peak
        DnaA total = bulk PD03831[c] + MONOMER0-160[c] + MONOMER0-4565[c]
                     (apo + ATP + ADP forms, counts)
  - in_band?       : cycle-means in [300,800] AND peak <= 800
  - divides?       : every steady gen completes a real division cycle
                     (>= STUB_MIN min and mass roughly doubles across the gen)
  - oriC pattern   : the set of oriC counts seen across the steady gens
                     (steady = {1,2}; reinitiation = a 4 appears)
  - steady?        : divides AND oriC never reaches 4

The figure draws, per seed, the DnaA-pool oscillation (trough-peak band + mean
marker) against the shaded [300,800] band, annotated with each seed's in-band?
and steady? verdict, and a bottom-line banner: does ANY seed achieve
in-band AND steady at V=1.2e-3.

Usage:
    python scripts/render_dnaa1_v12_multiseed.py \
        --run 0=out/dnaa1_v1p2e-3_s0/dnaa1_v1p2e-3_s0 \
        --run 1=out/dnaa1_v1p2e-3_s1/dnaa1_v1p2e-3_s1 \
        --run 2=out/dnaa1_v1p2e-3_s2/dnaa1_v1p2e-3_s2 \
        --run 3=out/dnaa1_v1p2e-3_s3/dnaa1_v1p2e-3_s3 \
        --steady-gens 4,5,6,7 --out studies/dnaa-1-expression/charts
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

try:
    from _run_provenance import run_id_from_run_dir
except Exception:  # pragma: no cover
    def run_id_from_run_dir(run_dir: str) -> str:
        return os.path.basename(run_dir.rstrip("/"))

DNAA_FORMS = ["PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"]
DNAA_CISTRON_IDX = 227
BAND_LO, BAND_HI = 300, 800
STUB_MIN = 5.0
V_LABEL = "1.2e-3"
DOUBLE_LO = 1.4   # min mass ratio (end/start) to call a generation a doubling


def per_gen_stats(run_dir: str) -> list[dict]:
    """Per-generation DnaA-pool envelope + init rate + oriC set + mass ratio."""
    files = glob.glob(os.path.join(run_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        files = glob.glob(os.path.join(run_dir, "**", "*.pq"), recursive=True)
    if not files:
        raise SystemExit(f"no history parquet under {run_dir}")

    ids = (pl.scan_parquet(files[0]).select("bulk__id").head(1)
           .collect()["bulk__id"][0].to_list())
    fi = [ids.index(m) for m in DNAA_FORMS]
    bc = pl.col("bulk__count")
    init_col = "listeners__rnap_data__rna_init_event_per_cistron"
    oric_col = "listeners__replication_data__number_of_oric"
    mass_col = "listeners__mass__cell_mass"
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(["generation", "global_time", mass_col, oric_col,
                   (bc.list.get(fi[0]) + bc.list.get(fi[1])
                    + bc.list.get(fi[2])).alias("dnaa_total"),
                   pl.col(init_col).list.get(DNAA_CISTRON_IDX, null_on_oob=True)
                     .fill_null(0).cast(pl.Int64).alias("dnaa_init")])
          .sort(["generation", "global_time"]).collect())

    dur = (df.group_by("generation")
             .agg(((pl.col("global_time").max() - pl.col("global_time").min())
                   / 60.0).alias("_dur")))
    real = sorted(dur.filter(pl.col("_dur") >= STUB_MIN)["generation"].to_list())

    out = []
    for gen in real:
        sub = df.filter(pl.col("generation") == gen).sort("global_time")
        t = sub["global_time"].to_numpy()
        ev = sub["dnaa_init"].to_numpy()
        mass = sub[mass_col].to_numpy()
        oric = sub[oric_col].to_numpy()
        d_min = max((t.max() - t.min()) / 60.0, 1e-9)
        mass_ratio = (float(mass[-1]) / float(mass[0])) if mass[0] > 0 else 0.0
        out.append(dict(
            gen=int(gen),
            dur_min=round(float(d_min), 1),
            dnaa_mean=round(float(sub["dnaa_total"].mean()), 1),
            dnaa_min=int(sub["dnaa_total"].min()),
            dnaa_max=int(sub["dnaa_total"].max()),
            init_per_min=round(float(ev.sum()) / d_min, 3),
            oric_set=sorted({int(v) for v in oric.tolist()}),
            oric_max=int(oric.max()),
            mass_ratio=round(mass_ratio, 2),
            divides=bool(mass_ratio >= DOUBLE_LO),
        ))
    return out


def steady_summary(stats: list[dict], steady_gens: list[int]) -> dict:
    sel = [s for s in stats if s["gen"] in steady_gens]
    if not sel:
        sel = stats[-min(4, len(stats)):]
    means = [s["dnaa_mean"] for s in sel]
    peak = max(s["dnaa_max"] for s in sel)
    trough = min(s["dnaa_min"] for s in sel)
    oric_union = sorted(set().union(*[set(s["oric_set"]) for s in sel]))
    oric_max = max(s["oric_max"] for s in sel)
    # A truncated TERMINAL generation (the run simply ended mid-cycle) is
    # incomplete, not a failed division — exclude it from the divides check
    # when its duration is well below the median of the steady gens.
    durs = [s["dur_min"] for s in sel]
    med_dur = float(np.median(durs)) if durs else 0.0
    div_sel = list(sel)
    if (len(div_sel) > 1 and div_sel[-1] is stats[-1]
            and div_sel[-1]["dur_min"] < 0.6 * med_dur):
        div_sel = div_sel[:-1]
    all_divide = all(s["divides"] for s in div_sel)
    in_band = all(BAND_LO <= m <= BAND_HI for m in means) and peak <= BAND_HI
    no_reinit = oric_max <= 2
    steady = bool(all_divide and no_reinit)
    return dict(
        steady_gens=[s["gen"] for s in sel],
        pool_trough=trough,
        pool_mean=float(np.mean(means)),
        pool_mean_lo=min(means),
        pool_mean_hi=max(means),
        pool_peak=peak,
        in_band=bool(in_band),
        all_divide=bool(all_divide),
        oric_union=oric_union,
        oric_max=oric_max,
        no_reinit=bool(no_reinit),
        steady=steady,
        mass_ratios=[s["mass_ratio"] for s in sel],
        divide_gens=[s["gen"] for s in div_sel],
        terminal_gen_truncated=bool(len(div_sel) < len(sel)),
    )


def render(runs: list[tuple[int, str]], out_dir: str,
           steady_gens: list[int]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(runs, key=lambda r: r[0])
    rows = []
    for seed, run_dir in runs:
        stats = per_gen_stats(run_dir)
        summ = steady_summary(stats, steady_gens)
        summ["seed"] = seed
        summ["run_dir"] = run_dir
        summ["run_id"] = run_id_from_run_dir(run_dir)
        summ["per_gen"] = stats
        rows.append(summ)
        print(f"seed {seed}: steady gens {summ['steady_gens']}  "
              f"pool trough/mean/peak = {summ['pool_trough']}/"
              f"{summ['pool_mean']:.0f}/{summ['pool_peak']}  "
              f"in_band={summ['in_band']}  divides={summ['all_divide']}  "
              f"oriC={summ['oric_union']} (max {summ['oric_max']})  "
              f"steady={summ['steady']}")

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axhspan(BAND_LO, BAND_HI, color="0.5", alpha=0.14, lw=0,
               label=f"accepted band [{BAND_LO}, {BAND_HI}]")

    xs = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi, mean = r["pool_trough"], r["pool_peak"], r["pool_mean"]
        col = "#2ca02c" if (r["in_band"] and r["steady"]) else (
              "#ff7f0e" if r["steady"] else "#d62728")
        # trough-peak oscillation band
        ax.add_patch(plt.Rectangle((i - 0.22, lo), 0.44, hi - lo,
                     facecolor=col, alpha=0.20, edgecolor=col, lw=1.2,
                     zorder=2))
        # trough & peak whisker caps
        ax.plot([i - 0.22, i + 0.22], [lo, lo], color=col, lw=1.4, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [hi, hi], color=col, lw=1.4, zorder=3)
        # mean marker
        ax.plot(i, mean, "o", color=col, ms=9, zorder=4)
        ax.annotate(f"{mean:.0f}", (i, mean), textcoords="offset points",
                    xytext=(10, -3), fontsize=8, color=col, fontweight="bold")
        ax.annotate(f"peak {hi}", (i, hi), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5, color=col)
        ax.annotate(f"trough {lo}", (i, lo), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7.5, color=col)
        # verdict text under the seed
        band_txt = "in-band" if r["in_band"] else "OVER band"
        steady_txt = "steady" if r["steady"] else "NOT steady"
        oric_txt = f"oriC {'↔'.join(map(str, r['oric_union']))}"
        ax.annotate(f"{band_txt}\n{steady_txt}\n{oric_txt}",
                    (i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {r['seed']}" for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    top = max(900, max(r["pool_peak"] for r in rows) * 1.08)
    ax.set_ylim(0, top)
    ax.set_ylabel("DnaA monomer pool (counts, all forms)\n"
                  "steady-gen trough–peak oscillation + cycle-mean")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.15)

    any_both = [r["seed"] for r in rows if r["in_band"] and r["steady"]]
    # closest: smallest peak overshoot among steady seeds, else among all
    steady_rows = [r for r in rows if r["steady"]] or rows
    closest = min(steady_rows, key=lambda r: max(0, r["pool_peak"] - BAND_HI)
                  + max(0, BAND_LO - r["pool_trough"]))
    if any_both:
        bottom = (f"BOTTOM LINE: seed(s) {any_both} achieve in-band AND "
                  f"steady-state at V={V_LABEL}.")
        bcol = "#2ca02c"
    else:
        bottom = (f"BOTTOM LINE: NO seed is in-band AND steady at V={V_LABEL} "
                  f"— closest is seed {closest['seed']} "
                  f"(peak {closest['pool_peak']} vs band hi {BAND_HI}, "
                  f"steady={closest['steady']}).")
        bcol = "#d62728"

    fig.suptitle(
        f"dnaa-1 — multiseed at V={V_LABEL}  (Mechanism A, seeds "
        f"{', '.join(str(r['seed']) for r in rows)}, steady gens {steady_gens})\n"
        "does the DnaA pool oscillate within band while staying steady-state?",
        fontsize=11, y=0.995)
    fig.text(0.5, 0.005, bottom, ha="center", va="bottom", fontsize=9.5,
             color=bcol, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa1_v12_multiseed")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "run_id": "dnaa1_v12_multiseed",
        "source_run_ids": [r["run_id"] for r in rows],
        "generation_id": None,
        "rendered_at": time.time(),
        "command": ("python scripts/render_dnaa1_v12_multiseed.py "
                    + " ".join(f"--run {r['seed']}={r['run_dir']}" for r in rows)
                    + f" --steady-gens {','.join(map(str, steady_gens))} "
                    + f"--out {out_dir}"),
        "V": 1.2e-3,
        "v_label": V_LABEL,
        "steady_gens": steady_gens,
        "band": [BAND_LO, BAND_HI],
        "double_ratio_lo": DOUBLE_LO,
        "any_seed_inband_and_steady": any_both,
        "closest_seed": closest["seed"],
        "bottom_line": bottom,
        "note": ("Multiseed at V=1.2e-3. Per seed, DnaA-pool steady-gen "
                 "trough/cycle-mean/peak vs band [300,800]; in_band = means in "
                 "band AND peak<=800; steady = every steady gen divides "
                 "(mass ratio>=1.4) AND oriC never reaches 4. DnaA total = bulk "
                 "PD03831[c]+MONOMER0-160[c]+MONOMER0-4565[c]; canonical "
                 "all-zeros daughter lineage; stub gens (<5 min) dropped."),
        "per_seed": [
            {k: r[k] for k in ("seed", "run_id", "steady_gens", "pool_trough",
                               "pool_mean", "pool_peak", "in_band", "all_divide",
                               "oric_union", "oric_max", "no_reinit", "steady",
                               "mass_ratios", "divide_gens",
                               "terminal_gen_truncated", "per_gen")}
            for r in rows
        ],
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"\nwrote {base}.svg / .png")
    print(f"  any seed in-band AND steady: {any_both or 'NONE'}")
    print(f"  closest seed: {closest['seed']}")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="SEED=RUNDIR")
    ap.add_argument("--steady-gens", default="4,5,6,7")
    ap.add_argument("--out", default="studies/dnaa-1-expression/charts")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run {spec!r} must be SEED=RUNDIR")
        ss, rd = spec.split("=", 1)
        runs.append((int(ss), rd))
    if not runs:
        raise SystemExit("need at least one --run SEED=RUNDIR")
    steady = [int(g) for g in a.steady_gens.split(",") if g.strip()]
    render(runs, a.out, steady)


if __name__ == "__main__":
    main()
