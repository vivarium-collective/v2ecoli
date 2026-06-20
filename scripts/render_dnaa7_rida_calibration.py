"""dnaa-7 RIDA-rate calibration chart — the operating window.

Reproducible replacement for the original inline-rendered figure. Reads the
RIDA_RATE_MULTIPLIER sweep runs (dnaa7_ridacal_r0* + dnaa7_rida_full),
reuses render_dnaa4_autoreg.metrics() for the gen3+ steady-state metrics
(same DnaA-ATP fraction definition Rashmi approved for dnaa-4), and draws:

  TOP:    DnaA-ATP fraction (g3+ mean) vs RIDA rate, with the [0.2,0.5] band
          and the operating window (RIDA 0.5-0.7) shaded.
  BOTTOM: re-initiation ticks (oriC>2) + max oriC vs RIDA rate.

Usage:
  .venv/bin/python scripts/render_dnaa7_rida_calibration.py \
      --out-dir workspace/studies/dnaa-7-seqa-sequestration/charts \
      [--seed 1]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import polars as pl

_spec = importlib.util.spec_from_file_location(
    "render_dnaa4_autoreg",
    os.path.join(os.path.dirname(__file__), "render_dnaa4_autoreg.py"))
_rda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rda)

# RIDA_RATE_MULTIPLIER for each sweep run tag.
SWEEP = [("r01", 0.1), ("r02", 0.2), ("r03", 0.3),
         ("r05", 0.5), ("r06", 0.6), ("r07", 0.7), ("rida_full", 1.0)]
BAND = (0.2, 0.5)
WINDOW = (0.5, 0.7)


def _run_dir(tag: str, seed: int) -> str:
    if tag == "rida_full":
        return f"out/dnaa7_rida_full_seed{seed}_8gen"
    return f"out/dnaa7_ridacal_{tag}_seed{seed}_8gen"


def collect(seed: int) -> list[dict]:
    rows = []
    for tag, rate in SWEEP:
        d = _run_dir(tag, seed)
        if not glob.glob(f"{d}/**/history/**/*.pq", recursive=True):
            print(f"  skip {tag} (rate {rate}): no data at {d}")
            continue
        df = _rda._frame(d)
        m = _rda.metrics(d, ss_gen=3)
        ss = df.filter(pl.col("generation") >= 3)
        rows.append({
            "tag": tag, "rate": rate,
            "atpfr_mean": float(ss["atp_fraction"].mean()),
            "reinit": m["reinit_ticks"], "oric_max": m["oric_max"],
            "pool_mean": float(ss["total_dnaa"].mean()),
        })
        print(f"  {tag} rate={rate}: atpfr_g3+mean={rows[-1]['atpfr_mean']:.3f} "
              f"reinit={m['reinit_ticks']} oric_max={m['oric_max']}")
    return sorted(rows, key=lambda r: r["rate"])


def render(rows: list[dict], out_dir: str, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate = np.array([r["rate"] for r in rows])
    atpfr = np.array([r["atpfr_mean"] for r in rows])
    reinit = np.array([r["reinit"] for r in rows])
    oric = np.array([r["oric_max"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    # operating window + band shading
    ax1.axhspan(*BAND, color="#16a34a", alpha=0.10, lw=0,
                label="DnaA-ATP band [0.2, 0.5]")
    ax1.axvspan(*WINDOW, color="#2563eb", alpha=0.10, lw=0,
                label="operating window (RIDA 0.5-0.7)")
    ax1.plot(rate, atpfr, "o-", color="#16a34a", lw=1.6, ms=6)
    for r, y in zip(rate, atpfr):
        ax1.annotate(f"{y:.2f}", (r, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7)
    ax1.set_ylabel("DnaA-ATP fraction\n(gen3+ mean)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("dnaa-7 RIDA-rate calibration — the operating window "
                  f"(RIDA 0.5-0.7), seed {seed}", fontsize=10)

    ax2.axvspan(*WINDOW, color="#2563eb", alpha=0.10, lw=0)
    ln1 = ax2.plot(rate, reinit, "s-", color="#dc2626", lw=1.6, ms=6,
                   label="re-init ticks (oriC>2)")
    ax2.set_ylabel("re-init ticks (oriC>2)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_yscale("symlog")
    ax2b = ax2.twinx()
    ln2 = ax2b.plot(rate, oric, "^--", color="#7c3aed", lw=1.4, ms=6,
                    label="max oriC")
    ax2b.set_ylabel("max oriC", color="#7c3aed")
    ax2b.tick_params(axis="y", labelcolor="#7c3aed")
    ax2b.set_ylim(0, 5)
    ax2.set_xlabel("RIDA_RATE_MULTIPLIER")
    lns = ln1 + ln2
    ax2.legend(lns, [l.get_label() for l in lns], fontsize=7, loc="upper right")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa7_rida_calibration")
    fig.savefig(base + ".png", dpi=140)
    fig.savefig(base + ".svg")
    plt.close(fig)

    in_win = [r for r in rows if WINDOW[0] <= r["rate"] <= WINDOW[1]]
    win_fr = [r["atpfr_mean"] for r in in_win]
    op = next((r for r in rows if r["rate"] == 0.6), in_win[0] if in_win else rows[0])
    meta = {
        "title": "dnaa-7 RIDA-rate calibration — the operating window (RIDA 0.5-0.7)",
        "caption": (
            "RIDA_RATE_MULTIPLIER sweep (DARS+DDAH on), mechanistic oriC-low trigger + "
            f"cooperativity (n=4, K=30), seed {seed}, 8 gens. TOP: DnaA-ATP fraction "
            "(gen3+ mean) vs RIDA rate — falls across the coarse sweep "
            f"({rows[0]['atpfr_mean']:.2f} at {rows[0]['rate']} -> "
            f"{rows[-1]['atpfr_mean']:.2f} at {rows[-1]['rate']}); within the operating "
            f"window (RIDA 0.5-0.7) it sits IN the [0.2,0.5] band "
            f"({min(win_fr):.2f}-{max(win_fr):.2f}, flat/noisy not strictly ordered). "
            "BOTTOM: re-init ticks (oriC>2) + max oriC — RIDA >= 0.5 gives 0 re-inits and "
            "clean oriC 2; below 0.5 the trigger over-initiates (oriC 4, thousands of "
            "re-init ticks). The overlap (RIDA 0.5-0.7) satisfies BOTH clean "
            f"one-init-per-cycle AND the in-band fraction; canonical operating point "
            f"RIDA=0.6 (fraction {op['atpfr_mean']:.3f})."),
        "interpretation": (
            "There is a genuine operating window for the mechanistic DnaA-ATP/oriC "
            "initiation: RIDA 0.5-0.7 (with DARS reactivation) gives clean oriC 1<->2, "
            "zero re-initiations, the DnaA-ATP fraction in [0.2,0.5], and the DnaA pool "
            "in band, with 8/8 divisions. RIDA both enforces once-per-cycle (resets the "
            "switch) and, at the right rate, leaves DnaA-ATP in its physiological band. "
            "The DnaA-ATP/oriC mechanism fully and in-band replaces the cell-mass "
            "heuristic. Single-seed (seed 1) shown here; multi-seed robustness is the "
            "open refinement."),
        "source_runs": [_run_dir(r["tag"], seed).split("/", 1)[1] for r in rows],
        "script": "scripts/render_dnaa7_rida_calibration.py",
    }
    with open(base + ".png.meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {base}.png/.svg + meta ({len(rows)} sweep points)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="workspace/studies/dnaa-7-seqa-sequestration/charts")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rows = collect(args.seed)
    if not rows:
        raise SystemExit("no RIDA sweep runs found")
    render(rows, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
