"""Real multi-replicate Wasserstein-2 comparison: PDMP-Phase-1 vs Phase-0 kFBA.

Extends ``compare_pdmp_vs_phase0.py`` (which ran ONE PDMP replicate and reported
a per-timepoint z-score) to a real ensemble Wasserstein gate:

  1. Run N >= 12 PDMP-Phase-1 replicates (``baseline_millard`` lqr=True with the
     ``consumption_matched`` reference-growth driver) at distinct master seeds,
     each for ``--duration`` simulated seconds.
  2. Load the committed Phase-0 kFBA reference ensemble from
     ``.pbg/runs/phase0-traj/seed_*``.
  3. Compute a per-observable 2-Wasserstein distance between the PDMP ensemble
     and the Phase-0 ensemble — endpoint distribution AND per-timepoint — for
     cell_mass and dry_mass, with bootstrap confidence intervals.

W2 vs scipy
-----------
``scipy.stats.wasserstein_distance`` computes the **1**-Wasserstein (W1)
distance, NOT W2. The pdmp-01 gate is specified as W2, so we compute the exact
one-dimensional 2-Wasserstein via the inverse-CDF (quantile) integral

    W2(a, b) = sqrt( ∫_0^1 ( F_a^{-1}(q) - F_b^{-1}(q) )^2 dq )

evaluated on a dense uniform quantile grid (closed form for 1-D empirical
distributions). We ALSO report scipy's W1 alongside, for cross-reference.

Honesty
-------
Nothing here decides "Accepted". The script reports the real W2 numbers, a
W2/σ effect size (W2 expressed in units of the Phase-0 endpoint SD), and a
mechanical gate verdict (``met`` iff endpoint W2 <= --gate-sigma * σ_phase0 for
BOTH observables). Any per-replicate crash, degenerate-gain warning, or
non-finite output is captured and recorded, never hidden.

Usage:
    .venv/bin/python scripts/compare_pdmp_ensemble_vs_phase0.py \
        --n-replicates 12 --duration 600 --sample-every 5 \
        --flux-source consumption_matched

Writes:
    .pbg/runs/pdmp-ensemble-vs-phase0/summary.json
    .pbg/runs/pdmp-ensemble-vs-phase0/replicate_<NN>.json
    reports/figures/pdmp-01/pdmp_vs_phase0.html   (overwrites the single-rep fig)
"""
from __future__ import annotations
import argparse
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance  # W1, for cross-reference

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

_DEGEN_KEYWORDS = ("degenerate", "singular", "zero gain", "riccati",
                   "negative values at equilibrium", "ill-conditioned")


# -------------------------- Phase-0 ensemble load ---------------------------

def load_phase0_ensemble(root: Path = Path(".pbg/runs/phase0-traj")):
    """Return (time[T], cell_mass[N,T], dry_mass[N,T]) for the committed ref."""
    seeds = sorted(root.glob("seed_*"))
    if not seeds:
        raise FileNotFoundError(f"No Phase-0 trajectories under {root}")
    rows_cell, rows_dry, t_ref, ref_len = [], [], None, None
    for seed_dir in seeds:
        traj_path = seed_dir / "trajectory.json"
        if not traj_path.is_file():
            continue
        traj = json.loads(traj_path.read_text())
        t = traj.get("time")
        cm = traj.get("listeners.mass.cell_mass")
        dm = traj.get("listeners.mass.dry_mass")
        if not (t and cm and dm):
            continue
        if t_ref is None:
            t_ref = np.asarray(t, dtype=float)
            ref_len = len(t_ref)
        if len(cm) != ref_len or len(dm) != ref_len:
            continue
        rows_cell.append(cm)
        rows_dry.append(dm)
    if not rows_cell:
        raise RuntimeError(f"No usable Phase-0 trajectories in {root}")
    return (t_ref,
            np.asarray(rows_cell, dtype=float),
            np.asarray(rows_dry, dtype=float))


# -------------------------- PDMP replicate ----------------------------------

def _to_float(v):
    if v is None:
        return np.nan
    mag = getattr(v, "magnitude", None)          # pint.Quantity
    if mag is not None:
        try:
            return float(mag)
        except (TypeError, ValueError):
            return np.nan
    as_number = getattr(v, "asNumber", None)     # unum.Unum
    if callable(as_number):
        try:
            return float(as_number())
        except (TypeError, ValueError):
            return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def run_pdmp_replicate(seed: int, duration_s: int, sample_every_s: int,
                       flux_source: str, out_dir: str) -> dict:
    """Run one PDMP replicate; capture trajectory + any degeneracy diagnostics.

    Top-level (Ray-picklable). Persists replicate_<seed>.json and returns a
    summary dict. Never raises: a crashed replicate returns error metadata so
    the ensemble can record partial coverage honestly.
    """
    import tempfile
    rec: dict = {"seed": seed, "flux_source": flux_source,
                 "duration_s": duration_s, "sample_every_s": sample_every_s}
    # Capture stdout/stderr to a REAL temp file (not StringIO) so library code
    # that calls sys.stdout.fileno() — some C extensions / progress bars do —
    # still works; we scan the captured text afterward for degeneracy markers.
    cap_file = tempfile.TemporaryFile(mode="w+")
    captured = ""
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(cap_file), contextlib.redirect_stderr(cap_file), \
                warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            from v2ecoli import build_composite
            c = build_composite(
                "ecoli_millard",
                lqr=True,
                seed=seed,
                with_ref_growth=True,
                ref_growth_flux_source=flux_source,
            )
            build_wall = time.perf_counter() - t0
            times, cell_masses, dry_masses = [], [], []
            sim_time = 0
            t_run = time.perf_counter()
            while sim_time < duration_s:
                c.run(sample_every_s)
                sim_time += sample_every_s
                mass = (c.state["agents"]["0"].get("listeners") or {}).get("mass") or {}
                times.append(sim_time)
                cell_masses.append(_to_float(mass.get("cell_mass")))
                dry_masses.append(_to_float(mass.get("dry_mass")))
            run_wall = time.perf_counter() - t_run
            warn_msgs = [str(w.message) for w in wlist]
    except Exception as e:  # noqa: BLE001 — record, never hide
        try:
            cap_file.seek(0); captured = cap_file.read()
        except Exception:
            pass
        finally:
            cap_file.close()
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
        rec["captured"] = captured[-2000:]
        return rec

    try:
        cap_file.seek(0); captured = cap_file.read()
    finally:
        cap_file.close()
    blob = (captured + "\n" + "\n".join(warn_msgs)).lower()
    degen_hits = sorted({k for k in _DEGEN_KEYWORDS if k in blob})

    rec.update({
        "time": times,
        "cell_mass": cell_masses,
        "dry_mass": dry_masses,
        "build_wall_s": round(build_wall, 2),
        "run_wall_s": round(run_wall, 2),
        "cell_final": cell_masses[-1] if cell_masses else None,
        "dry_final": dry_masses[-1] if dry_masses else None,
        "degeneracy_keywords": degen_hits,
        "n_warnings": len(warn_msgs),
    })
    Path(out_dir, f"replicate_{seed:02d}.json").write_text(json.dumps(rec))
    return rec


# -------------------------- Wasserstein-2 -----------------------------------

def w2_1d(a, b, n_q: int = 512) -> float:
    """Exact 1-D 2-Wasserstein between two empirical samples via the quantile
    (inverse-CDF) integral. Same physical units as the inputs."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = (np.arange(n_q) + 0.5) / n_q
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def w1_1d(a, b) -> float:
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def bootstrap_w2_ci(a, b, n_boot: int = 1000, n_q: int = 512,
                    boot_seed: int = 12345):
    """Percentile bootstrap 95% CI for W2 by resampling each ensemble with
    replacement. Deterministic given boot_seed."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(boot_seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        vals[i] = w2_1d(aa, bb, n_q)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


# -------------------------- run the ensemble --------------------------------

def run_pdmp_ensemble(seeds, duration_s, sample_every_s, flux_source,
                      out_dir, parallel="sequential", num_threads=None):
    run_kwargs = dict(duration_s=duration_s, sample_every_s=sample_every_s,
                      flux_source=flux_source, out_dir=str(out_dir))
    if parallel == "ray":
        from v2ecoli.library.parallel_seeds import run_seeds_parallel
        pr = run_seeds_parallel(seeds, run_pdmp_replicate, mode="ray",
                                run_kwargs=run_kwargs, num_threads=num_threads)
        recs = [r for r in pr.results if r is not None]
        print(f"  Ray fan-out: threads/worker={pr.n_threads_per_worker} "
              f"crit-path={pr.wall_s}s")
        return recs
    recs = []
    for s in seeds:
        print(f"  PDMP replicate seed={s} ...", flush=True)
        r = run_pdmp_replicate(s, **run_kwargs)
        if "error" in r:
            print(f"    -> ERROR {r['error']}", flush=True)
        else:
            cf, df = r.get("cell_final"), r.get("dry_final")
            tag = (f"cell_final={cf:.1f} dry_final={df:.1f} "
                   f"build={r.get('build_wall_s')}s run={r.get('run_wall_s')}s"
                   if cf is not None and df is not None else "no-endpoint")
            if r.get("degeneracy_keywords"):
                tag += f"  DEGEN={r['degeneracy_keywords']}"
            print(f"    -> {tag}", flush=True)
        recs.append(r)
    return recs


def stack_pdmp(recs, t_ref):
    """Stack successful replicate trajectories, interpolated onto the Phase-0
    time grid t_ref so both ensembles share one axis.

    The Phase-0 runner snapshots at t=[1,6,...,600] (warm-up tick + stride),
    while the PDMP loop samples at t=[5,10,...,600]; the grids differ in offset
    and length, so we resample each PDMP replicate onto t_ref via linear
    interpolation. The endpoint (t=600) is present in both grids exactly, so
    the endpoint W2 uses the measured final value, not an extrapolation.
    """
    good = [r for r in recs if "error" not in r and r.get("cell_mass")
            and r.get("time")]
    cell, dry = [], []
    for r in good:
        tt = np.asarray(r["time"], dtype=float)
        cm = np.asarray(r["cell_mass"], dtype=float)
        dm = np.asarray(r["dry_mass"], dtype=float)
        cell.append(np.interp(t_ref, tt, cm))
        dry.append(np.interp(t_ref, tt, dm))
    cell = np.asarray(cell, dtype=float) if cell else np.empty((0, len(t_ref)))
    dry = np.asarray(dry, dtype=float) if dry else np.empty((0, len(t_ref)))
    return cell, dry, good


# -------------------------- viz ---------------------------------------------

def fig_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_fig(t, cm_p0, dm_p0, cm_pd, dm_pd, w2t_cell, w2t_dry):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def overlay(ax, p0, pd, ylabel, title):
        m0, s0 = np.nanmean(p0, 0), np.nanstd(p0, 0)
        ax.fill_between(t, m0 - s0, m0 + s0, alpha=0.22, color="#1f77b4",
                        label=f"Phase-0 kFBA (N={p0.shape[0]}, ±1σ)")
        ax.plot(t, m0, color="#1f77b4", lw=1.4)
        for i in range(pd.shape[0]):
            ax.plot(t, pd[i], color="#d62728", alpha=0.18, lw=0.7)
        mp = np.nanmean(pd, 0)
        ax.plot(t, mp, color="#d62728", lw=2.0,
                label=f"PDMP-Phase-1 (N={pd.shape[0]} mean)")
        ax.set_xlabel("sim time (s)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    overlay(axes[0, 0], cm_p0, cm_pd, "cell_mass (fg)",
            "cell_mass(t) — PDMP ensemble vs Phase-0")
    overlay(axes[0, 1], dm_p0, dm_pd, "dry_mass (fg)",
            "dry_mass(t) — PDMP ensemble vs Phase-0")

    ax = axes[1, 0]
    ax.plot(t, w2t_cell, color="#7c3aed", lw=1.8, label="cell_mass")
    ax.plot(t, w2t_dry, color="#0891b2", lw=1.8, label="dry_mass")
    ax.set_xlabel("sim time (s)"); ax.set_ylabel("W₂(t) (fg)")
    ax.set_title("Per-timepoint W₂ distance (PDMP vs Phase-0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(cm_p0[:, -1], bins=12, alpha=0.5, color="#1f77b4",
            label="Phase-0 cell_mass (endpt)")
    ax.hist(cm_pd[:, -1], bins=12, alpha=0.5, color="#d62728",
            label="PDMP cell_mass (endpt)")
    ax.set_xlabel("cell_mass @ endpoint (fg)"); ax.set_ylabel("count")
    ax.set_title("Endpoint cell_mass distributions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("PDMP-Phase-1 ensemble vs Phase-0 kFBA — real W₂ gate "
                 "(M9-glucose, consumption_matched)", fontsize=12)
    fig.tight_layout()
    return fig


def write_html(out_path: Path, data_uri: str, S: dict):
    def fnum(x, p=2):
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
            else f"{x:.{p}f}"
    gate = S["gate"]
    verdict_color = "#15803d" if gate["met"] else "#b91c1c"
    verdict_text = "MET" if gate["met"] else "NOT met"
    degen = S.get("degeneracy_any")
    degen_html = ""
    if degen:
        degen_html = (f'<p class="ref" style="color:#b45309"><b>Degeneracy '
                      f'flag:</b> {degen} replicate(s) emitted a degenerate/'
                      f'singular-gain or equilibrium warning — see summary.json '
                      f'<code>degeneracy_keywords</code>.</p>')
    rows = ""
    for obs in ("cell_mass", "dry_mass"):
        e = S["endpoint"][obs]
        rows += (
            f'<tr><td>{obs} (fg)</td>'
            f'<td class="num">{fnum(e["w2"])}</td>'
            f'<td class="num">[{fnum(e["w2_ci"][0])}, {fnum(e["w2_ci"][1])}]</td>'
            f'<td class="num">{fnum(e["w1"])}</td>'
            f'<td class="num">{fnum(e["sigma_phase0"])}</td>'
            f'<td class="num" style="font-weight:600">{fnum(e["w2_over_sigma"])}</td>'
            f'<td class="num">{fnum(e["pdmp_mean"])}</td>'
            f'<td class="num">{fnum(e["phase0_mean"])}</td></tr>')
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>PDMP-Phase-1 ensemble vs Phase-0 — real W₂</title>
<link rel="stylesheet" href="../../assets/style.css">
<style>
  body {{ margin: 0; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 22px; }}
  table.cmp {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 0.92em; }}
  table.cmp th, table.cmp td {{ padding: 6px 10px; border: 1px solid var(--border); }}
  table.cmp th {{ background: rgba(0,0,0,0.03); }}
  table.cmp td.num {{ text-align: right; font-variant-numeric: tabular-nums;
                       font-family: ui-monospace, Menlo, monospace; }}
  img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 4px; }}
  .verdict {{ font-weight: 700; color: {verdict_color}; }}
</style></head>
<body>
<header>
  <h1>PDMP-Phase-1 ensemble vs Phase-0 kFBA — real Wasserstein-2</h1>
  <p class="subtitle">M9-glucose · PDMP N={S["n_pdmp"]} (consumption_matched) vs
     Phase-0 N={S["n_phase0"]} · {S["duration_s"]}s · auto-generated {S["date"]}</p>
</header>
<main>
<p>
This replaces the earlier single-replicate z-score figure with a real
multi-replicate <b>2-Wasserstein</b> gate. We ran {S["n_pdmp_requested"]}
PDMP-Phase-1 replicates ({S["n_pdmp"]} usable) at distinct master seeds with the
<code>consumption_matched</code> reference-growth driver, and compare their
cell_mass / dry_mass distributions against the committed Phase-0 kFBA reference
ensemble (N={S["n_phase0"]}). W₂ is the exact 1-D 2-Wasserstein via the
inverse-CDF integral (units = fg); W₁ (scipy) is shown for cross-reference.
<b>W₂/σ</b> expresses the endpoint W₂ in units of the Phase-0 endpoint standard
deviation — a scale-free effect size.
</p>
<p>Gate (mechanical): endpoint W₂ ≤ {S["gate"]["sigma_mult"]:g}·σ<sub>Phase-0</sub>
for BOTH observables → <span class="verdict">{verdict_text}</span>.
{S["gate"]["detail"]}</p>
{degen_html}
<table class="cmp">
  <tr><th>observable</th><th>W₂ (endpt)</th><th>W₂ 95% CI</th><th>W₁ (scipy)</th>
      <th>σ Phase-0</th><th>W₂/σ</th><th>PDMP mean</th><th>Phase-0 mean</th></tr>
  {rows}
</table>
<p class="ref">PDMP replicate wall: build {fnum(S["timing"]["build_mean_s"],1)}s,
run {fnum(S["timing"]["run_mean_s"],1)}s (mean) for {S["duration_s"]} simulated
seconds each; ensemble wall {fnum(S["timing"]["ensemble_wall_s"],1)}s
({S["parallel"]}). Phase-0 ensemble from
<code>.pbg/runs/phase0-traj/seed_*</code>; PDMP replicates from
<code>.pbg/runs/pdmp-ensemble-vs-phase0/replicate_*.json</code>.</p>
<img src="{data_uri}" alt="PDMP ensemble vs Phase-0 W2 panel">
<p class="ref" style="margin-top:16px">
Honest status: this figure reports the real W₂; it does not by itself mark the
study Accepted. The mechanical verdict above is <span class="verdict">{verdict_text}</span>.
</p>
</main>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


# -------------------------- main --------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-replicates", type=int, default=12)
    p.add_argument("--seed-start", type=int, default=1000,
                   help="first PDMP master seed (distinct from Phase-0 0..N).")
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--flux-source", default="consumption_matched",
                   choices=["proportional", "measured_kfba", "consumption_matched"])
    p.add_argument("--phase0-root", default=".pbg/runs/phase0-traj")
    p.add_argument("--out-dir", default=".pbg/runs/pdmp-ensemble-vs-phase0")
    p.add_argument("--viz-out", default="reports/figures/pdmp-01/pdmp_vs_phase0.html")
    p.add_argument("--parallel", choices=("ray", "sequential"), default="sequential")
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--gate-sigma", type=float, default=2.0,
                   help="endpoint W2 must be <= this * Phase-0 endpoint SD to 'meet'.")
    p.add_argument("--date", default="", help="ISO date stamp for the figure.")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_out = REPO_ROOT / args.viz_out
    viz_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase-0 reference from {args.phase0_root}...")
    t_p0, cm_p0, dm_p0 = load_phase0_ensemble(Path(args.phase0_root))
    ref_len = len(t_p0)
    print(f"  Phase-0: N={cm_p0.shape[0]} × {ref_len} timepoints "
          f"({t_p0[0]:.0f}..{t_p0[-1]:.0f}s)")

    seeds = list(range(args.seed_start, args.seed_start + args.n_replicates))
    print(f"\nRunning {len(seeds)} PDMP replicates "
          f"({args.flux_source}, {args.duration}s, seeds {seeds[0]}..{seeds[-1]})...")
    t_ens = time.perf_counter()
    recs = run_pdmp_ensemble(seeds, args.duration, args.sample_every,
                             args.flux_source, out_dir,
                             parallel=args.parallel, num_threads=args.num_threads)
    ensemble_wall = time.perf_counter() - t_ens

    cm_pd, dm_pd, good = stack_pdmp(recs, t_p0)
    n_pdmp = cm_pd.shape[0]
    n_err = sum(1 for r in recs if "error" in r)
    degen_any = sum(1 for r in recs if r.get("degeneracy_keywords"))
    print(f"\nPDMP usable replicates: {n_pdmp}/{len(seeds)} "
          f"(errors={n_err}, degeneracy-flagged={degen_any})")
    if n_pdmp == 0:
        # Still emit a summary so the failure is a committed artifact.
        summary = {"n_phase0": int(cm_p0.shape[0]), "n_pdmp": 0,
                   "n_pdmp_requested": len(seeds), "n_errors": n_err,
                   "errors": [r.get("error") for r in recs if "error" in r],
                   "ensemble_wall_s": round(ensemble_wall, 1)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        sys.exit("All PDMP replicates failed; wrote failure summary, no figure.")

    # W2 / W1 — endpoint + per-timepoint.
    endpoint = {}
    for obs, p0, pd in (("cell_mass", cm_p0, cm_pd), ("dry_mass", dm_p0, dm_pd)):
        a_end, b_end = p0[:, -1], pd[:, -1]
        w2 = w2_1d(b_end, a_end)
        w1 = w1_1d(b_end, a_end)
        ci = bootstrap_w2_ci(b_end, a_end)
        sigma0 = float(np.nanstd(a_end))
        endpoint[obs] = {
            "w2": w2, "w2_ci": list(ci), "w1": w1,
            "sigma_phase0": sigma0,
            "w2_over_sigma": (w2 / sigma0) if sigma0 > 0 else float("nan"),
            "pdmp_mean": float(np.nanmean(b_end)),
            "pdmp_std": float(np.nanstd(b_end)),
            "phase0_mean": float(np.nanmean(a_end)),
        }

    w2t_cell = np.array([w2_1d(cm_pd[:, i], cm_p0[:, i]) for i in range(ref_len)])
    w2t_dry = np.array([w2_1d(dm_pd[:, i], dm_p0[:, i]) for i in range(ref_len)])

    met = all(np.isfinite(endpoint[o]["w2_over_sigma"])
              and endpoint[o]["w2_over_sigma"] <= args.gate_sigma
              for o in ("cell_mass", "dry_mass"))
    gate = {
        "sigma_mult": args.gate_sigma, "met": bool(met),
        "detail": (f'cell_mass W₂/σ={endpoint["cell_mass"]["w2_over_sigma"]:.1f}, '
                   f'dry_mass W₂/σ={endpoint["dry_mass"]["w2_over_sigma"]:.1f}.'),
    }

    builds = [r["build_wall_s"] for r in good if "build_wall_s" in r]
    runs = [r["run_wall_s"] for r in good if "run_wall_s" in r]
    summary = {
        "date": args.date or "unstamped",
        "condition": "M9-glucose",
        "flux_source": args.flux_source,
        "duration_s": args.duration,
        "sample_every_s": args.sample_every,
        "n_phase0": int(cm_p0.shape[0]),
        "n_pdmp": int(n_pdmp),
        "n_pdmp_requested": len(seeds),
        "n_errors": int(n_err),
        "degeneracy_any": int(degen_any),
        "degeneracy_keywords": sorted({k for r in recs
                                       for k in r.get("degeneracy_keywords", [])}),
        "pdmp_seeds": seeds,
        "endpoint": endpoint,
        "w2t_cell_mass_max": float(np.nanmax(w2t_cell)),
        "w2t_dry_mass_max": float(np.nanmax(w2t_dry)),
        "gate": gate,
        "parallel": args.parallel,
        "timing": {
            "build_mean_s": float(np.mean(builds)) if builds else None,
            "run_mean_s": float(np.mean(runs)) if runs else None,
            "ensemble_wall_s": round(ensemble_wall, 1),
        },
        "errors": [r.get("error") for r in recs if "error" in r],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir/'summary.json'}")
    for obs in ("cell_mass", "dry_mass"):
        e = endpoint[obs]
        print(f"  {obs}: endpoint W2={e['w2']:.2f} fg "
              f"(95% CI [{e['w2_ci'][0]:.2f},{e['w2_ci'][1]:.2f}]) "
              f"W1={e['w1']:.2f}  W2/σ={e['w2_over_sigma']:.2f}")
    print(f"  GATE (W2 ≤ {args.gate_sigma}σ both obs): "
          f"{'MET' if met else 'NOT met'}")

    fig = make_fig(t_p0, cm_p0, dm_p0, cm_pd, dm_pd, w2t_cell, w2t_dry)
    write_html(viz_out, fig_to_data_uri(fig), summary)
    print(f"Wrote {viz_out}")


if __name__ == "__main__":
    main()
