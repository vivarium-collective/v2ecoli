"""Phase 4 sprint 3 — column-centric runtime prototype.

Sprint 2 identified the marshalling hotspots: ~14K isinstance/tick,
4K __array_finalize__/tick, 52 process_update/tick, etc. The
column-centric runtime ELIMINATES these by replacing nested-dict
state with pre-allocated arrays indexed by (trajectory, time).

This sprint demonstrates the architectural win with a minimal toy
Poisson-sampler that mimics the Phase-3 jump-process work:

  Per tick:
    rates = n_active * promoter_init_probs   # shape (N_promoters,)
    k ~ Poisson(rates)                       # the observable
    log_lik = sum(log P(k | rates))

Two runners do the SAME math, different state representations:

  A. pbg-style: a real pbg Composite with one Step doing the sampling
     per trajectory, dict-based state. Run N trajectories sequentially.
  B. column-centric: pure numpy arrays shape (N_traj, N_promoters)
     for the latest tick. Vector ops across the trajectory dim.

Benchmarks across N ∈ {1, 4, 16, 64, 256, 1024}. Wall time per
trajectory should be ~constant for column-centric and ~linear in N
for pbg (linear because each trajectory pays the framework tax).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Match the WCM order of magnitude so the toy is representative.
N_PROMOTERS = 3277
DEFAULT_N_ACTIVE = 1000
DEFAULT_T_TICKS = 60


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def collect_provenance():
    try:
        sha = _git("rev-parse", "HEAD")
    except Exception:
        sha = "(unknown)"
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        branch = "(unknown)"
    try:
        subprocess.run(["git", "update-index", "--really-refresh"],
                       check=False, capture_output=True)
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"],
                           capture_output=True)
        dirty = r.returncode != 0
    except Exception:
        dirty = False
    return {"sha": sha, "short": sha[:8] if sha != "(unknown)" else sha,
            "branch": branch, "dirty": dirty,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version()}


def make_promoter_probs(rng) -> np.ndarray:
    p = rng.dirichlet(np.ones(N_PROMOTERS))
    return p


# ---------------------------------------------------------------------------
# Runner A — pbg-style: real Composite, dict-based state, N sequential runs.
# ---------------------------------------------------------------------------

def run_pbg_style(n_traj: int, t_ticks: int, n_active: int,
                  probs: np.ndarray) -> tuple[float, np.ndarray]:
    """N independent pbg Composites, each running the toy model for t_ticks.

    Uses v2ecoli.core.build_core() to get a fully-wired core that
    matches the production runtime — same path-discovery, same type
    system, same emit infrastructure as the WCM composites.
    """
    from process_bigraph import Composite, Process
    from v2ecoli.core import build_core

    class ToyPoissonStep(Process):
        config_schema = {
            "n_active": "integer",
            "probs_list": "list[float]",
            "seed": "integer",
        }

        def initialize(self, config):
            self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
            self.n_active = int(self.config["n_active"])
            self.probs = np.asarray(self.config["probs_list"],
                                    dtype=np.float64)

        def inputs(self):
            return {"log_lik": "float"}

        def outputs(self):
            return {"log_lik": "float"}

        def update(self, state, interval):
            rates = self.n_active * self.probs
            k = self.rng.poisson(rates).astype(np.int64)
            log_lik = float(poisson.logpmf(k, rates).sum())
            return {"log_lik": log_lik}

    results = np.zeros((n_traj, t_ticks))
    t0 = time.perf_counter()
    for traj in range(n_traj):
        core = build_core()
        core.register_link("ToyPoisson", ToyPoissonStep)
        doc = {
            "toy": {
                "_type": "process",
                "address": "local:ToyPoisson",
                "config": {
                    "n_active": n_active,
                    "probs_list": probs.tolist(),
                    "seed": traj,
                },
                "inputs": {"log_lik": ["log_lik_store"]},
                "outputs": {"log_lik": ["log_lik_store"]},
            },
            "log_lik_store": 0.0,
        }
        comp = Composite({"state": doc}, core=core)
        for t in range(t_ticks):
            comp.run(1)
            results[traj, t] = float(comp.state.get("log_lik_store", 0.0))
    wall = time.perf_counter() - t0
    return wall, results


# ---------------------------------------------------------------------------
# Runner B — column-centric: pure numpy vector ops over the trajectory dim.
# ---------------------------------------------------------------------------

def run_column_centric(n_traj: int, t_ticks: int, n_active: int,
                       probs: np.ndarray) -> tuple[float, np.ndarray]:
    """Single numpy run; all N trajectories advance in lockstep per tick."""
    rng = np.random.default_rng(0)
    rates = n_active * probs  # shape (N_promoters,)
    results = np.zeros((n_traj, t_ticks))
    t0 = time.perf_counter()
    for t in range(t_ticks):
        k = rng.poisson(rates, size=(n_traj, N_PROMOTERS))  # (N_traj, N_p)
        log_lik = poisson.logpmf(k, rates).sum(axis=1)  # (N_traj,)
        results[:, t] = log_lik
    wall = time.perf_counter() - t0
    return wall, results


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def benchmark(n_trajs: list[int], t_ticks: int, n_active: int):
    rng = np.random.default_rng(42)
    probs = make_promoter_probs(rng)

    results = []
    for n in n_trajs:
        print(f"\nN={n:>5} trajectories × {t_ticks} ticks "
              f"× {N_PROMOTERS} promoters")
        wall_cc, res_cc = run_column_centric(n, t_ticks, n_active, probs)
        per_traj_cc = wall_cc / n * 1000
        print(f"  column-centric: {wall_cc * 1000:9.1f} ms total, "
              f"{per_traj_cc:7.3f} ms/trajectory")

        # pbg-style: skip for large N to keep this sprint short.
        if n <= 64:
            wall_pbg, _ = run_pbg_style(n, t_ticks, n_active, probs)
            per_traj_pbg = wall_pbg / n * 1000
            print(f"  pbg-style:      {wall_pbg * 1000:9.1f} ms total, "
                  f"{per_traj_pbg:7.3f} ms/trajectory")
            speedup = per_traj_pbg / per_traj_cc
            print(f"  column-centric speedup: {speedup:6.1f}×")
        else:
            wall_pbg = None
            per_traj_pbg = None
            speedup = None

        results.append({
            "n": n,
            "wall_cc_ms": wall_cc * 1000,
            "per_traj_cc_ms": per_traj_cc,
            "wall_pbg_ms": wall_pbg * 1000 if wall_pbg else None,
            "per_traj_pbg_ms": per_traj_pbg,
            "speedup": speedup,
        })
    return results


def make_figure(results) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ns = np.array([r["n"] for r in results])
    cc_per = np.array([r["per_traj_cc_ms"] for r in results])
    pbg_ns = np.array([r["n"] for r in results
                       if r["per_traj_pbg_ms"] is not None])
    pbg_per = np.array([r["per_traj_pbg_ms"] for r in results
                        if r["per_traj_pbg_ms"] is not None])

    # Panel 1: per-trajectory wall vs N (log-log).
    ax = axes[0]
    if len(pbg_per) > 0:
        ax.loglog(pbg_ns, pbg_per, "o-", color="#dc2626", lw=2,
                  markersize=10, label="pbg-style (dict state, sequential)")
    ax.loglog(ns, cc_per, "o-", color="#16a34a", lw=2, markersize=10,
              label="column-centric (numpy, vectorized)")
    ax.set_xlabel("Number of trajectories (N)")
    ax.set_ylabel("ms per trajectory (60 ticks × 3277 promoters)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title("Wall time per trajectory — column-centric is "
                 "(near-)constant in N",
                 loc="left", fontsize=11, fontweight="bold")

    # Panel 2: total wall vs N.
    ax = axes[1]
    cc_total = np.array([r["wall_cc_ms"] for r in results])
    pbg_total = np.array([r["wall_pbg_ms"] for r in results
                          if r["wall_pbg_ms"] is not None])
    if len(pbg_total) > 0:
        ax.loglog(pbg_ns, pbg_total, "o-", color="#dc2626", lw=2,
                  markersize=10, label="pbg-style")
    ax.loglog(ns, cc_total, "o-", color="#16a34a", lw=2, markersize=10,
              label="column-centric")
    # Reference: linear (ideal pbg scaling), constant (ideal CC)
    ax.loglog(ns, cc_total[0] * np.ones_like(ns), "k:", alpha=0.5,
              label="ideal constant total")
    ax.loglog(ns, cc_total[0] * ns / ns[0], "k--", alpha=0.5,
              label="ideal linear total (pbg)")
    ax.set_xlabel("Number of trajectories (N)")
    ax.set_ylabel("Total wall time (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Total wall vs N",
                 loc="left", fontsize=11, fontweight="bold")

    fig.suptitle("Phase-4 sprint 3: column-centric runtime prototype",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, results, t_ticks, n_active, plot_uri):
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    rows = []
    for r in results:
        pbg_cell = (f"{r['per_traj_pbg_ms']:.2f}"
                    if r["per_traj_pbg_ms"] is not None
                    else "<em>skipped (N too large)</em>")
        speedup_cell = (f"{r['speedup']:.1f}×"
                       if r["speedup"] is not None
                       else "<em>—</em>")
        rows.append(
            f"<tr><td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['per_traj_cc_ms']:.3f}</td>"
            f"<td class='num'>{pbg_cell}</td>"
            f"<td class='num'>{speedup_cell}</td>"
            f"<td class='num'>{r['wall_cc_ms']:.1f}</td>"
            f"<td class='num'>{r['wall_pbg_ms']:.1f}"
            if r["wall_pbg_ms"] else f"<td class='num'>—</td>"
            f"</td></tr>")

    # Projection to N=1000.
    cc_at_1024 = next((r for r in results if r["n"] >= 1024), None)
    if cc_at_1024:
        proj = (f"At N=1024 trajectories, column-centric finishes "
                f"<strong>{cc_at_1024['wall_cc_ms']:.0f} ms total</strong> "
                f"(<strong>{cc_at_1024['per_traj_cc_ms']:.3f} ms/trajectory</strong>).")
    else:
        proj = ""

    table_rows = "\n".join(rows)
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 4 sprint 3 — column-centric runtime prototype</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1500px; margin: 24px auto; padding: 0 18px;
         line-height: 1.55; }}
  h1 {{ margin: 0 0 6px 0; color:#1e3a8a; }}
  h2 {{ margin-top: 24px; border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px; color:#1e3a8a; }}
  .provenance {{ background:#f8fafc; border:1px solid #e2e8f0;
                 border-radius:8px; padding:10px 14px; margin:14px 0 20px;
                 font-size:0.85em; }}
  .provenance dt {{ display:inline-block; min-width:110px; color:#475569;
                    font-weight:600; }}
  .provenance dd {{ display:inline; margin:0;
                    font-family: ui-monospace, Menlo, monospace; }}
  .provenance .row {{ margin: 1px 0; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%;
           font-size: 0.92em; }}
  th, td {{ padding: 6px 12px; border: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  td.num {{ text-align: right;
            font-family: ui-monospace, Menlo, monospace; }}
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px; }}
  .takeaway {{ background:#dcfce7; border-left:4px solid #16a34a;
               padding:12px 16px; margin:14px 0; font-size:0.95em; }}
</style>

<h1>Phase 4 sprint 3 — column-centric runtime prototype</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Toy Poisson sampler mimicking the Phase-3 transcript-initiation
  step ({N_PROMOTERS} promoters, n_active={n_active}, {t_ticks} ticks).
  Two runners: pbg-style (dict state, sequential N runs) vs
  column-centric (numpy arrays, vectorized over N trajectories).
</p>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{prov['generated']}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{prov['sha']}"
       style="color:#0369a1;text-decoration:none">{prov['short']}</a>
       &nbsp;<code>{prov['sha']}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{prov['branch']}</dd></div>
  <div class="row"><dt>host</dt><dd>{prov['host']} &nbsp;
    <span style="color:#94a3b8">Python {prov['python']}</span></dd></div>
</div>

<h2>Scaling — wall time per trajectory vs N</h2>
<img class="plot" src="{plot_uri}" alt="wall vs N comparison">

<h2>Numbers</h2>
<table>
  <tr><th>N</th>
      <th>CC ms/traj</th>
      <th>pbg ms/traj</th>
      <th>CC speedup</th>
      <th>CC total ms</th>
      <th>pbg total ms</th></tr>
  {table_rows}
</table>

<div class="takeaway">
  <strong>Architectural win.</strong> {proj} The pbg-style runner
  pays the framework tax (composite.run, process_update, ~14K
  isinstance/tick, ~4K __array_finalize__/tick) on EVERY trajectory.
  The column-centric runner does the same math but with one numpy
  Poisson call across all N trajectories per tick — the framework
  overhead is amortized once over the entire batch.
</div>

<h2>Phase 4 implication</h2>
<p>
  This is the proof-of-concept for the Phase-4 architectural pivot
  signaled by sprints 1-2. Sprint 4 candidate: extend this prototype
  to a multi-Step composite shape (transcript initiation + polypeptide
  initiation + likelihood collector — the Phase-3 stack), keeping
  state as pre-allocated arrays and measuring whether the speedup
  curve holds up under realistic process composition.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trajs", default="1,4,16,64,256,1024",
                    help="Comma-separated trajectory counts to sweep.")
    ap.add_argument("--t-ticks", type=int, default=DEFAULT_T_TICKS)
    ap.add_argument("--n-active", type=int, default=DEFAULT_N_ACTIVE)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-04/column_centric_prototype.html"))
    args = ap.parse_args()

    n_trajs = [int(x) for x in args.n_trajs.split(",")]

    results = benchmark(n_trajs, args.t_ticks, args.n_active)

    print("\nSummary:")
    for r in results:
        print(f"  N={r['n']:>5}: CC={r['per_traj_cc_ms']:7.3f} ms/traj   "
              f"pbg={r['per_traj_pbg_ms'] or '---':>7} ms/traj   "
              f"speedup={r['speedup'] or '---'}×")

    plot_uri = make_figure(results)
    write_html(args.out, results, args.t_ticks, args.n_active, plot_uri)


if __name__ == "__main__":
    main()
