"""Measure DnaA / cell-cycle observables over a multigen run — the single
canonical probe for the dnaa-replication investigation.

Originally Rashmi's per-generation mean-DnaA probe (PR #97). Extended to cover
the whole investigation's acceptance checks so there is ONE analysis tool:

  * DnaA monomer trough / cycle-mean / peak per generation, and whether the
    WHOLE oscillation (not just the mean) sits in a band — for dnaa-1's
    "oscillate within [300,800]" tuning (Rashmi 2026-05-31).
  * DnaA concentration (nM), dnaA mRNA, and dnaA mRNA initiation rate
    (events/min, dnaA TU index 2700) — Rashmi's original metrics.
  * oriC periodicity (max <= 2 and visits {1,2} from a steady-state gen) and
    cell_mass periodicity (cross-gen CV) — for dnaa-0 acceptance.
  * DnaA-ATP fraction (apo / DnaA-ATP / DnaA-ADP) vs the Boesen [0.2,0.5] band —
    for dnaa-2, when the dnaa_cycle listener is present.

Columns that aren't in a given run (oriC, dnaA_cycle, ...) are simply skipped,
so the same tool works for dnaa-0, dnaa-1 and dnaa-2 runs.

Usage::

    .venv/bin/python scripts/measure_dnaa_at_steady_state.py \\
        --run-dir studies/dnaa-1-expression-dynamics/parquet-runs/<exp-id>/<exp-id> \\
        --experiment-id <exp-id> [--gens 1-7] [--steady-from 3] \\
        [--band 300 800] [--atp-band 0.2 0.5] [--lineage-seed 0]

``--run-dir`` is the dir that contains the ``history/`` hive (the runner nests
it under <out-dir>/<experiment-id>/, so pass that inner dir).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import polars as pl

DNAA_MONOMER_IDX = 3861   # PD03831[c] aggregate (total DnaA across forms)
DNAA_TU_IDX = 2700        # dnaA in rna_counts.mRNA_counts and rnap_data.rna_init_event
AVOGADRO = 6.022e23

# (parquet column, attribute) — read only those present in the run.
_OPTIONAL_COLS = [
    "global_time",
    "listeners__monomer_counts",
    "listeners__mass__volume",
    "listeners__mass__cell_mass",
    "listeners__rna_counts__mRNA_counts",
    "listeners__rnap_data__rna_init_event",
    "listeners__replication_data__number_of_oric",
    "listeners__dnaA_cycle__atp_fraction",
    "listeners__dnaA_cycle__apo_count",
    "listeners__dnaA_cycle__atp_count",
    "listeners__dnaA_cycle__adp_count",
]


def _gen_dirs(run_dir: str, exp_id: str, seed: int) -> dict[int, str]:
    """Map generation -> the followed-daughter (agent_id='0'*gen) parquet glob.
    Locates the history hive under run_dir wherever it is nested."""
    hist = glob.glob(f"{run_dir}/**/history/experiment_id={exp_id}",
                     recursive=True)
    if not hist:
        hist = glob.glob(f"{run_dir}/history/experiment_id={exp_id}")
    if not hist:
        return {}
    base = f"{hist[0]}/variant=0/lineage_seed={seed}"
    out = {}
    for d in sorted(glob.glob(f"{base}/generation=*")):
        g = int(d.split("generation=")[-1])
        aid = "0" * g
        pat = f"{d}/agent_id={aid}/*.pq"
        if glob.glob(pat):
            out[g] = pat
    return out


def _read_gen(pat: str) -> pl.DataFrame | None:
    files = sorted(glob.glob(pat),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not files:
        return None
    have = pl.read_parquet(files[0]).columns
    cols = [c for c in _OPTIONAL_COLS if c in have]
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files]).sort(
        "global_time")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="dir containing the history/ hive")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--gens", default=None,
                    help="generation range 'a-b' (default: all present)")
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--steady-from", type=int, default=3)
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 800.0],
                    metavar=("LOW", "HIGH"), help="DnaA total band")
    ap.add_argument("--atp-band", type=float, nargs=2, default=[0.2, 0.5],
                    metavar=("LOW", "HIGH"), help="DnaA-ATP fraction band")
    args = ap.parse_args()

    gen_pats = _gen_dirs(args.run_dir, args.experiment_id, args.lineage_seed)
    if not gen_pats:
        print(f"ERROR: no parquet history under {args.run_dir} "
              f"(experiment_id={args.experiment_id}, seed={args.lineage_seed})")
        return 1
    if args.gens:
        a, b = (int(x) for x in args.gens.split("-"))
        gen_pats = {g: p for g, p in gen_pats.items() if a <= g <= b}

    band_lo, band_hi = args.band
    atp_lo, atp_hi = args.atp_band
    rows = []
    print(f"run: {args.run_dir} (exp {args.experiment_id}, seed {args.lineage_seed})")
    for g in sorted(gen_pats):
        df = _read_gen(gen_pats[g])
        if df is None:
            continue
        r = {"gen": g, "n": df.height}
        t = df["global_time"].to_numpy()
        r["dur"] = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 0.0
        if "listeners__monomer_counts" in df.columns:
            d = df["listeners__monomer_counts"].list.get(DNAA_MONOMER_IDX).to_numpy().astype(float)
            r.update(trough=float(d.min()), mean=float(d.mean()), peak=float(d.max()))
        if "listeners__mass__cell_mass" in df.columns:
            m = df["listeners__mass__cell_mass"].to_numpy().astype(float)
            r.update(birth_mass=float(m[0]), peak_mass=float(m.max()))
        if "listeners__rnap_data__rna_init_event" in df.columns and r["dur"] > 0:
            inits = df["listeners__rnap_data__rna_init_event"].list.get(DNAA_TU_IDX).fill_null(0).to_numpy()
            r["init_rate"] = float(inits.sum() / r["dur"])
        if "listeners__replication_data__number_of_oric" in df.columns:
            oric = df["listeners__replication_data__number_of_oric"].to_numpy()
            r["oric_set"] = sorted(set(int(x) for x in oric))
            r["oric_max"] = int(oric.max())
        if "listeners__dnaA_cycle__atp_fraction" in df.columns:
            af = df["listeners__dnaA_cycle__atp_fraction"].drop_nulls().to_numpy()
            if len(af):
                r["atp_frac_mean"] = float(af.mean())
                r["atp_frac_end"] = float(af[-1])
        rows.append(r)
        seg = [f"gen {g}: n={r['n']:5d} τ={r['dur']:5.1f}m"]
        if "mean" in r:
            seg.append(f"DnaA trough/mean/peak={r['trough']:.0f}/{r['mean']:.0f}/{r['peak']:.0f}")
        if "init_rate" in r:
            seg.append(f"init={r['init_rate']:.2f}/min")
        if "oric_max" in r:
            seg.append(f"oriC={r['oric_set']}")
        if "atp_frac_mean" in r:
            seg.append(f"ATP-frac={r['atp_frac_mean']:.3f}")
        print("  " + "  ".join(seg))

    steady = [r for r in rows if r["gen"] >= args.steady_from]
    if not steady:
        print("no steady-state generations in range"); return 1
    print(f"\nSteady state (gens {args.steady_from}+):")
    checks = []

    if all("mean" in r for r in steady):
        troughs = np.array([r["trough"] for r in steady])
        peaks = np.array([r["peak"] for r in steady])
        means = np.array([r["mean"] for r in steady])
        cv = float(means.std() / means.mean()) if means.mean() else 1.0
        whole = bool((troughs >= band_lo).all() and (peaks <= band_hi).all())
        mean_in = bool((means >= band_lo).all() and (means <= band_hi).all())
        print(f"  DnaA: trough {troughs.min():.0f}-{troughs.max():.0f}, "
              f"peak {peaks.min():.0f}-{peaks.max():.0f}, "
              f"cycle-mean {means.min():.0f}-{means.max():.0f} (CV {cv*100:.1f}%)")
        print(f"    whole-oscillation in [{band_lo:.0f},{band_hi:.0f}]: "
              f"{'YES' if whole else 'NO'}  |  cycle-mean in band: "
              f"{'YES' if mean_in else 'NO'}")
        checks.append(("DnaA mean in band", mean_in))
    if all("birth_mass" in r for r in steady):
        births = np.array([r["birth_mass"] for r in steady])
        peaksm = np.array([r["peak_mass"] for r in steady])
        cvb = float(births.std() / births.mean()) if births.mean() else 1.0
        cvp = float(peaksm.std() / peaksm.mean()) if peaksm.mean() else 1.0
        ok = cvb < 0.05 and cvp < 0.05
        print(f"  cell_mass: CV birth {cvb:.3f} / peak {cvp:.3f}  "
              f"(periodic if < 0.05): {'YES' if ok else 'NO'}")
        checks.append(("cell_mass periodic", ok))
    if all("init_rate" in r for r in steady):
        rates = np.array([r["init_rate"] for r in steady])
        print(f"  dnaA init rate: median {np.median(rates):.2f}/min "
              f"(target ~1/min)")
    if all("oric_max" in r for r in steady):
        union = set().union(*[set(r["oric_set"]) for r in steady])
        ok = all(r["oric_max"] <= 2 for r in steady) and {1, 2} <= union
        print(f"  oriC: periodic 1<->2 (max<=2, visits {{1,2}}) from gen "
              f"{args.steady_from}: {'YES' if ok else 'NO'}")
        checks.append(("oriC periodic 1<->2", ok))
    if all("atp_frac_mean" in r for r in steady):
        afm = np.array([r["atp_frac_mean"] for r in steady])
        ok = bool((afm >= atp_lo).all() and (afm <= atp_hi).all())
        print(f"  DnaA-ATP fraction: {afm.min():.3f}-{afm.max():.3f} "
              f"(target [{atp_lo},{atp_hi}]): {'YES' if ok else 'NO'}")
        checks.append(("DnaA-ATP fraction in band", ok))

    if checks:
        allok = all(ok for _, ok in checks)
        print(f"\n  CHECKS: " + "; ".join(
            f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in checks))
        return 0 if allok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
