"""dnaa-1 CHOOSE-THE-V sweep figure (Rashmi's request, 2026-06-08).

Plots the dnaa-1 Mechanism-A V-sweep AS A FUNCTION OF V so the reviewer can
pick the best value. Two stacked panels share the V (log) x-axis:

  (top)    DnaA monomer pool at steady state — per-V trough / cycle-mean / peak
           across the steady generations (default gens 4-7), drawn as a marker
           (cycle-mean) with a [trough, peak] error band. The accepted
           literature band [300, 800] is shaded; the in-band-pool V region is
           annotated.
  (bottom) dnaA mRNA initiation rate per V — per-V mean over the steady gens
           with a [min, max] error band, the ~1/min biological target line and
           its +/-50% tolerance band shaded; the on-target-rate V region is
           annotated.

The current pick (0.8e-3) is marked, and the V where BOTH criteria are best
jointly satisfied is highlighted.

Each V's per-generation stats are computed with the SAME definitions as
scripts/render_dnaa1_decision.py:
  total DnaA  = sum of the three bulk forms PD03831[c] + MONOMER0-160[c]
                + MONOMER0-4565[c]  (counts)
  init rate   = dnaA cistron (idx 227) rna_init_event_per_cistron summed over a
                generation / generation duration (events/min)
on the canonical all-zeros daughter lineage, daughter-stub gens (<5 min)
dropped.

Usage:
    python scripts/render_dnaa1_v_sweep.py \\
        --run 0.8e-3=out/dnaa1_v0p8e-3/dnaa1_v0p8e-3 \\
        --run 0.9e-3=out/dnaa1_v0p9e-3/dnaa1_v0p9e-3 \\
        --run 1.0e-3=out/dnaa1_v1p0e-3/dnaa1_v1p0e-3 \\
        --run 1.1e-3=out/dnaa1_v1p1e-3/dnaa1_v1p1e-3 \\
        --run 1.7e-3=out/dnaa1_v1p7e-3/dnaa1_v1p7e-3 \\
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
RATE_TARGET = 1.0
RATE_TOL = 0.5          # +/-50% tolerance band around the target
STUB_MIN = 5.0


def per_gen_stats(run_dir: str) -> list[dict]:
    """Per-generation DnaA-pool envelope (trough/mean/peak) + init rate."""
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
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(["generation", "global_time",
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
        sub = df.filter(pl.col("generation") == gen)
        t = sub["global_time"].to_numpy()
        ev = sub["dnaa_init"].to_numpy()
        d_min = max((t.max() - t.min()) / 60.0, 1e-9)
        out.append(dict(
            gen=int(gen),
            dur_min=round(float(d_min), 1),
            dnaa_mean=round(float(sub["dnaa_total"].mean()), 1),
            dnaa_min=int(sub["dnaa_total"].min()),
            dnaa_max=int(sub["dnaa_total"].max()),
            init_per_min=round(float(ev.sum()) / d_min, 3),
        ))
    return out


def steady_envelope(stats: list[dict], steady_gens: list[int]) -> dict:
    """Collapse per-gen stats over the steady gens into one envelope."""
    sel = [s for s in stats if s["gen"] in steady_gens]
    if not sel:
        sel = stats[-min(4, len(stats)):]  # fallback: last up-to-4 gens
    means = [s["dnaa_mean"] for s in sel]
    rates = [s["init_per_min"] for s in sel]
    return dict(
        steady_gens=[s["gen"] for s in sel],
        pool_trough=min(s["dnaa_min"] for s in sel),
        pool_mean=float(np.mean(means)),
        pool_mean_lo=min(means),
        pool_mean_hi=max(means),
        pool_peak=max(s["dnaa_max"] for s in sel),
        rate_mean=float(np.mean(rates)),
        rate_min=min(rates),
        rate_max=max(rates),
        in_band=all(BAND_LO <= m <= BAND_HI for m in means)
                and max(s["dnaa_max"] for s in sel) <= BAND_HI,
        on_target=all(abs(r - RATE_TARGET) <= RATE_TOL * RATE_TARGET
                      for r in rates),
    )


def render(runs: list[tuple[float, str]], out_dir: str,
           steady_gens: list[int], current_pick: float | None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(runs, key=lambda r: r[0])
    rows = []
    for v, run_dir in runs:
        stats = per_gen_stats(run_dir)
        env = steady_envelope(stats, steady_gens)
        env["V"] = v
        env["run_dir"] = run_dir
        env["run_id"] = run_id_from_run_dir(run_dir)
        env["per_gen"] = stats
        rows.append(env)
        print(f"V={v:.1e}: steady gens {env['steady_gens']}  "
              f"pool trough/mean/peak = {env['pool_trough']}/"
              f"{env['pool_mean']:.0f}/{env['pool_peak']}  "
              f"in_band={env['in_band']}  "
              f"rate {env['rate_min']:.2f}-{env['rate_max']:.2f} "
              f"(mean {env['rate_mean']:.2f})/min  on_target={env['on_target']}")

    V = np.array([r["V"] for r in rows])
    pool_mean = np.array([r["pool_mean"] for r in rows])
    pool_lo = np.array([r["pool_trough"] for r in rows])
    pool_hi = np.array([r["pool_peak"] for r in rows])
    rate_mean = np.array([r["rate_mean"] for r in rows])
    rate_lo = np.array([r["rate_min"] for r in rows])
    rate_hi = np.array([r["rate_max"] for r in rows])

    fig, ax = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # ---- (top) DnaA pool vs V ----
    a = ax[0]
    a.axhspan(BAND_LO, BAND_HI, color="0.5", alpha=0.13, lw=0,
              label=f"accepted band [{BAND_LO}, {BAND_HI}]")
    a.errorbar(V, pool_mean,
               yerr=[pool_mean - pool_lo, pool_hi - pool_mean],
               fmt="o-", color="#222222", ecolor="#888888", elinewidth=1.4,
               capsize=4, ms=7, lw=1.6, zorder=4,
               label="steady-gen cycle-mean (bars: trough–peak)")
    for r in rows:
        a.annotate(f"{r['pool_mean']:.0f}", (r["V"], r["pool_mean"]),
                   textcoords="offset points", xytext=(7, 6),
                   fontsize=8, color="#222222")
    # shade the in-band-pool V region
    in_band_V = [r["V"] for r in rows if r["in_band"]]
    if in_band_V:
        a.axvspan(min(in_band_V) * 0.94, max(in_band_V) * 1.06,
                  color="#2ca02c", alpha=0.08, lw=0, zorder=1)
        a.annotate("pool in band", (np.sqrt(min(in_band_V) * max(in_band_V)),
                   BAND_HI), color="#2ca02c", fontsize=9, ha="center",
                   va="bottom", fontweight="bold")
    a.set_ylabel("DnaA monomer pool\n(counts, all forms)")
    a.set_title("DnaA pool at steady state vs V  (in-band region shaded green)",
                fontsize=10)
    a.legend(loc="upper left", fontsize=8, framealpha=0.92)
    a.grid(True, alpha=0.15)

    # ---- (bottom) init rate vs V ----
    b = ax[1]
    b.axhspan(RATE_TARGET * (1 - RATE_TOL), RATE_TARGET * (1 + RATE_TOL),
              color="#1f77b4", alpha=0.10, lw=0,
              label=f"target ±{int(RATE_TOL*100)}% "
                    f"[{RATE_TARGET*(1-RATE_TOL):.1f}, {RATE_TARGET*(1+RATE_TOL):.1f}]/min")
    b.axhline(RATE_TARGET, color="#1f77b4", lw=1.3, ls="--", alpha=0.8,
              label="biological target ≈1/min")
    b.errorbar(V, rate_mean,
               yerr=[rate_mean - rate_lo, rate_hi - rate_mean],
               fmt="s-", color="#8c564b", ecolor="#c4a59c", elinewidth=1.4,
               capsize=4, ms=7, lw=1.6, zorder=4,
               label="steady-gen mean (bars: min–max)")
    for r in rows:
        b.annotate(f"{r['rate_mean']:.2f}", (r["V"], r["rate_mean"]),
                   textcoords="offset points", xytext=(7, -12),
                   fontsize=8, color="#8c564b")
    on_target_V = [r["V"] for r in rows if r["on_target"]]
    if on_target_V:
        b.axvspan(min(on_target_V) * 0.94, max(on_target_V) * 1.06,
                  color="#1f77b4", alpha=0.08, lw=0, zorder=1)
        b.annotate("rate on target", (np.sqrt(min(on_target_V) * max(on_target_V)),
                   RATE_TARGET * (1 + RATE_TOL)), color="#1f77b4", fontsize=9,
                   ha="center", va="bottom", fontweight="bold")
    b.set_ylabel("dnaA mRNA init rate\n(events/min)")
    b.set_xlabel("V  =  TU00259[c] dnaA synth-prob (perturbation)")
    b.set_ylim(bottom=0)
    b.set_title("dnaA initiation rate vs V  (on-target region shaded blue)",
                fontsize=10)
    b.legend(loc="upper left", fontsize=8, framealpha=0.92)
    b.grid(True, alpha=0.15)

    a.set_xscale("log")
    a.set_xticks(V)
    a.set_xticklabels([f"{v*1e3:g}e-3" for v in V])
    a.minorticks_off()

    # mark current pick + best-joint V on both panels
    both_V = [r["V"] for r in rows if r["in_band"] and r["on_target"]]
    for axis in ax:
        if current_pick is not None:
            axis.axvline(current_pick, color="#d62728", lw=1.1, ls="-.",
                         alpha=0.65, zorder=2)
        for bv in both_V:
            axis.axvline(bv, color="#2ca02c", lw=2.0, alpha=0.5, zorder=2)
    if current_pick is not None:
        ax[0].annotate("current pick", (current_pick, ax[0].get_ylim()[1]),
                       color="#d62728", fontsize=8, ha="center", va="top",
                       rotation=90, xytext=(-6, -4),
                       textcoords="offset points")

    joint_note = (f"both criteria met at V={', '.join(f'{v*1e3:g}e-3' for v in both_V)}"
                  if both_V else
                  "NO single V meets both criteria — band-pool and ~1/min-rate "
                  "are in tension (both rise with V)")
    fig.suptitle(
        "dnaa-1 — choose-the-V sweep (Mechanism A, seed 1, 7 gens, "
        f"steady gens {steady_gens})\n{joint_note}",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa1_v_sweep")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "run_id": "dnaa1_v_sweep",
        "source_run_ids": [r["run_id"] for r in rows],
        "generation_id": None,
        "rendered_at": time.time(),
        "command": ("python scripts/render_dnaa1_v_sweep.py "
                    + " ".join(f"--run {r['V']:g}={r['run_dir']}" for r in rows)
                    + f" --steady-gens {','.join(map(str, steady_gens))} "
                    + f"--out {out_dir}"),
        "steady_gens": steady_gens,
        "band": [BAND_LO, BAND_HI],
        "rate_target": RATE_TARGET,
        "rate_tol_frac": RATE_TOL,
        "current_pick": current_pick,
        "both_criteria_V": both_V,
        "note": ("Choose-the-V decision figure. (top) DnaA pool trough/cycle-"
                 "mean/peak per V over the steady gens, band [300,800] shaded; "
                 "(bottom) dnaA mRNA init rate per V, ~1/min target +/-50% band. "
                 "DnaA total = sum of bulk PD03831[c]+MONOMER0-160[c]+"
                 "MONOMER0-4565[c]; init = cistron 227 per-gen events/duration. "
                 "Canonical all-zeros daughter lineage; stub gens (<5 min) "
                 "dropped. Each V applied via run-time --perturbation override."),
        "per_V": [
            {k: r[k] for k in ("V", "run_id", "steady_gens", "pool_trough",
                               "pool_mean", "pool_peak", "in_band", "rate_mean",
                               "rate_min", "rate_max", "on_target", "per_gen")}
            for r in rows
        ],
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"\nwrote {base}.svg / .png")
    print(f"  both-criteria V: {both_V or 'none (tension)'}")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="V=RUNDIR",
                    help="V_value=run_dir, e.g. 0.9e-3=out/dnaa1_v0p9e-3/dnaa1_v0p9e-3")
    ap.add_argument("--steady-gens", default="4,5,6,7",
                    help="comma-separated steady generation indices")
    ap.add_argument("--current-pick", type=float, default=0.8e-3)
    ap.add_argument("--out", default="studies/dnaa-1-expression/charts")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run {spec!r} must be V=RUNDIR")
        vs, rd = spec.split("=", 1)
        runs.append((float(vs), rd))
    if not runs:
        raise SystemExit("need at least one --run V=RUNDIR")
    steady = [int(g) for g in a.steady_gens.split(",") if g.strip()]
    render(runs, a.out, steady, a.current_pick)


if __name__ == "__main__":
    main()
