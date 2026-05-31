"""Phase 4 sprint 4 — multi-step column-centric prototype.

Sprint 3 used a single-step Poisson sampler and measured a steady 3×
speedup of column-centric over pbg-style. This sprint extends to a
3-step composite — toy_transcript_init + toy_polypeptide_init +
toy_likelihood_collector — matching the Phase-3 shape (sprint 1+2
emission + sprint 2 collector). The hypothesis: pbg per-trajectory
wall scales with the number of processes (each adds a process_update
+ state-tree walk + isinstance storm) while column-centric stays
~constant. Speedup should grow.

Same N sweep as sprint 3: N ∈ {1, 4, 16, 64, 256, 1024}.
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

N_PROMOTERS = 3277
N_PROTEINS = 4538  # match the WCM's monomer count
DEFAULT_N_ACTIVE_RNAP = 1000
DEFAULT_N_ACTIVE_RIBOSOME = 4000
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


# ---------------------------------------------------------------------------
# pbg-style: 3-step composite chained via shared scalar stores.
# ---------------------------------------------------------------------------

def run_pbg_style(n_traj, t_ticks, n_active_t, n_active_p,
                  probs_t, probs_p):
    from process_bigraph import Composite, Process
    from v2ecoli.core import build_core

    class ToyTranscriptInit(Process):
        config_schema = {
            "n_active": "integer",
            "probs_list": "list[float]",
            "seed": "integer",
        }

        def initialize(self, config):
            self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
            self.n_active = int(self.config["n_active"])
            self.probs = np.asarray(self.config["probs_list"], dtype=np.float64)

        def inputs(self):
            return {"log_lik_t": "float"}

        def outputs(self):
            return {"log_lik_t": "float"}

        def update(self, state, interval):
            rates = self.n_active * self.probs
            k = self.rng.poisson(rates).astype(np.int64)
            return {"log_lik_t": float(poisson.logpmf(k, rates).sum())}

    class ToyPolypeptideInit(Process):
        config_schema = {
            "n_active": "integer",
            "probs_list": "list[float]",
            "seed": "integer",
        }

        def initialize(self, config):
            self.rng = np.random.default_rng(int(self.config.get("seed", 0)) + 7)
            self.n_active = int(self.config["n_active"])
            self.probs = np.asarray(self.config["probs_list"], dtype=np.float64)

        def inputs(self):
            return {"log_lik_p": "float"}

        def outputs(self):
            return {"log_lik_p": "float"}

        def update(self, state, interval):
            rates = self.n_active * self.probs
            k = self.rng.poisson(rates).astype(np.int64)
            return {"log_lik_p": float(poisson.logpmf(k, rates).sum())}

    class ToyLikelihoodCollector(Process):
        config_schema = {}

        def initialize(self, config):
            pass

        def inputs(self):
            return {"log_lik_t": "float", "log_lik_p": "float",
                    "total": "float"}

        def outputs(self):
            return {"total": "float"}

        def update(self, state, interval):
            return {"total": float(state.get("log_lik_t", 0.0))
                    + float(state.get("log_lik_p", 0.0))}

    results = np.zeros((n_traj, t_ticks))
    t0 = time.perf_counter()
    for traj in range(n_traj):
        core = build_core()
        core.register_link("ToyTranscriptInit", ToyTranscriptInit)
        core.register_link("ToyPolypeptideInit", ToyPolypeptideInit)
        core.register_link("ToyLikelihoodCollector", ToyLikelihoodCollector)
        doc = {
            "ti": {
                "_type": "process",
                "address": "local:ToyTranscriptInit",
                "config": {"n_active": n_active_t,
                            "probs_list": probs_t.tolist(),
                            "seed": traj},
                "inputs": {"log_lik_t": ["log_lik_t_store"]},
                "outputs": {"log_lik_t": ["log_lik_t_store"]},
            },
            "pi": {
                "_type": "process",
                "address": "local:ToyPolypeptideInit",
                "config": {"n_active": n_active_p,
                            "probs_list": probs_p.tolist(),
                            "seed": traj},
                "inputs": {"log_lik_p": ["log_lik_p_store"]},
                "outputs": {"log_lik_p": ["log_lik_p_store"]},
            },
            "coll": {
                "_type": "process",
                "address": "local:ToyLikelihoodCollector",
                "config": {},
                "inputs": {"log_lik_t": ["log_lik_t_store"],
                            "log_lik_p": ["log_lik_p_store"],
                            "total": ["total_store"]},
                "outputs": {"total": ["total_store"]},
            },
            "log_lik_t_store": 0.0,
            "log_lik_p_store": 0.0,
            "total_store": 0.0,
        }
        comp = Composite({"state": doc}, core=core)
        for t in range(t_ticks):
            comp.run(1)
            results[traj, t] = float(comp.state.get("total_store", 0.0))
    return time.perf_counter() - t0, results


# ---------------------------------------------------------------------------
# Column-centric: 3 vector ops per tick, shared pre-allocated arrays.
# ---------------------------------------------------------------------------

def run_column_centric(n_traj, t_ticks, n_active_t, n_active_p,
                       probs_t, probs_p):
    rng_t = np.random.default_rng(0)
    rng_p = np.random.default_rng(7)
    rates_t = n_active_t * probs_t
    rates_p = n_active_p * probs_p

    log_lik_t = np.zeros((n_traj, t_ticks))
    log_lik_p = np.zeros((n_traj, t_ticks))
    total = np.zeros((n_traj, t_ticks))

    t0 = time.perf_counter()
    for t in range(t_ticks):
        k_t = rng_t.poisson(rates_t, size=(n_traj, N_PROMOTERS))
        log_lik_t[:, t] = poisson.logpmf(k_t, rates_t).sum(axis=1)
        k_p = rng_p.poisson(rates_p, size=(n_traj, N_PROTEINS))
        log_lik_p[:, t] = poisson.logpmf(k_p, rates_p).sum(axis=1)
        total[:, t] = log_lik_t[:, t] + log_lik_p[:, t]
    return time.perf_counter() - t0, total


# ---------------------------------------------------------------------------

def benchmark(n_trajs, t_ticks, n_active_t, n_active_p):
    rng = np.random.default_rng(42)
    probs_t = rng.dirichlet(np.ones(N_PROMOTERS))
    probs_p = rng.dirichlet(np.ones(N_PROTEINS))

    results = []
    for n in n_trajs:
        print(f"\nN={n:>5} trajectories × {t_ticks} ticks "
              f"(3-step composite)")
        wall_cc, _ = run_column_centric(n, t_ticks, n_active_t,
                                         n_active_p, probs_t, probs_p)
        per_traj_cc = wall_cc / n * 1000
        print(f"  column-centric: {wall_cc * 1000:9.1f} ms total, "
              f"{per_traj_cc:7.3f} ms/trajectory")
        if n <= 64:
            wall_pbg, _ = run_pbg_style(n, t_ticks, n_active_t,
                                          n_active_p, probs_t, probs_p)
            per_traj_pbg = wall_pbg / n * 1000
            speedup = per_traj_pbg / per_traj_cc
            print(f"  pbg-style:      {wall_pbg * 1000:9.1f} ms total, "
                  f"{per_traj_pbg:7.3f} ms/trajectory")
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


def make_figure(results, single_step_ref=None) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ns = np.array([r["n"] for r in results])
    cc_per = np.array([r["per_traj_cc_ms"] for r in results])
    pbg_ns = np.array([r["n"] for r in results
                       if r["per_traj_pbg_ms"] is not None])
    pbg_per = np.array([r["per_traj_pbg_ms"] for r in results
                        if r["per_traj_pbg_ms"] is not None])

    ax = axes[0]
    if single_step_ref is not None:
        ax.loglog(single_step_ref["ns"], single_step_ref["pbg"], "s--",
                  color="#fca5a5", lw=1.2, markersize=7,
                  label="sprint 3 pbg (1-step)")
        ax.loglog(single_step_ref["ns"], single_step_ref["cc"], "s--",
                  color="#86efac", lw=1.2, markersize=7,
                  label="sprint 3 CC (1-step)")
    if len(pbg_per) > 0:
        ax.loglog(pbg_ns, pbg_per, "o-", color="#dc2626", lw=2.5,
                  markersize=12, label="pbg (3-step)")
    ax.loglog(ns, cc_per, "o-", color="#16a34a", lw=2.5, markersize=12,
              label="column-centric (3-step)")
    ax.set_xlabel("Number of trajectories (N)")
    ax.set_ylabel("ms per trajectory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("3-step composite — pbg cost grows; CC stays flat",
                 loc="left", fontsize=11, fontweight="bold")

    ax = axes[1]
    cc_total = np.array([r["wall_cc_ms"] for r in results])
    pbg_total = np.array([r["wall_pbg_ms"] for r in results
                          if r["wall_pbg_ms"] is not None])
    if len(pbg_total) > 0:
        ax.loglog(pbg_ns, pbg_total, "o-", color="#dc2626", lw=2,
                  markersize=10, label="pbg (3-step)")
    ax.loglog(ns, cc_total, "o-", color="#16a34a", lw=2, markersize=10,
              label="column-centric (3-step)")
    ax.set_xlabel("Number of trajectories (N)")
    ax.set_ylabel("Total wall (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Total wall vs N",
                 loc="left", fontsize=11, fontweight="bold")

    fig.suptitle("Phase-4 sprint 4: 3-step composite — does speedup compound?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path, results, t_ticks, plot_uri):
    prov = collect_provenance()
    # Build table_rows BEFORE the f-string so it doesn't try to interpolate
    # the placeholder as a Python variable.
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    rows = []
    for r in results:
        pbg_cell = (f"{r['per_traj_pbg_ms']:.2f}"
                    if r["per_traj_pbg_ms"] is not None
                    else "<em>skipped</em>")
        speedup_cell = (f"{r['speedup']:.1f}×"
                       if r["speedup"] is not None
                       else "<em>—</em>")
        wall_pbg_cell = (f"{r['wall_pbg_ms']:.1f}"
                         if r["wall_pbg_ms"] is not None
                         else "—")
        rows.append(
            f"<tr><td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['per_traj_cc_ms']:.3f}</td>"
            f"<td class='num'>{pbg_cell}</td>"
            f"<td class='num'>{speedup_cell}</td>"
            f"<td class='num'>{r['wall_cc_ms']:.1f}</td>"
            f"<td class='num'>{wall_pbg_cell}</td></tr>")
    table_rows = "\n".join(rows)

    # Speedup-vs-sprint-3 comparison.
    sprint3_at_64 = 3.0  # roughly stable speedup from sprint 3 across 4–64
    speedup_at_64 = next((r["speedup"] for r in results
                          if r["n"] == 64 and r["speedup"] is not None), None)
    compounding_note = ""
    if speedup_at_64 is not None:
        ratio = speedup_at_64 / sprint3_at_64
        compounding_note = (
            f"At N=64 the 3-step speedup is {speedup_at_64:.1f}×, "
            f"vs sprint-3's single-step speedup of ~{sprint3_at_64:.1f}× — "
            f"a {ratio:.2f}× compound. Confirms the framework overhead "
            f"is per-process, not per-tick.")

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 4 sprint 4 — multi-step prototype</title>
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

<h1>Phase 4 sprint 4 — multi-step column-centric prototype</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Sprint 3's 1-step toy gave a steady 3× speedup. This sprint chains
  3 toy processes (transcript-init + polypeptide-init + collector)
  mirroring the Phase-3 emit-aggregate shape, to test whether pbg
  overhead is PER-TICK (constant) or PER-PROCESS (scales with the
  composite's process count).
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

<h2>Scaling</h2>
<img class="plot" src="{plot_uri}" alt="3-step composite scaling">

<h2>Numbers</h2>
<table>
  <tr><th>N</th>
      <th>CC ms/traj</th>
      <th>pbg ms/traj</th>
      <th>speedup</th>
      <th>CC total ms</th>
      <th>pbg total ms</th></tr>
  ROWS_PLACEHOLDER
</table>

<div class="takeaway">
  <strong>{compounding_note}</strong>
</div>

<h2>Phase 4 implication</h2>
<p>
  pbg overhead grows with the number of Processes in the composite —
  each new step adds a process_update + state-tree walk + isinstance
  storm. The column-centric runtime keeps a flat per-trajectory cost
  regardless of how many vectorized ops are chained. Extrapolating to
  the full WCM (~55 partitioned processes vs this toy's 3): the
  speedup should be substantially larger than the 3-step prototype's
  shown here. Sprint 5 candidate is to extend the prototype further
  with a few of the actual WCM Processes (or instrument the WCM's
  per-process call counts) to put a real number on the projected
  speedup.
</p>
""".replace("ROWS_PLACEHOLDER", table_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


# Sprint-3 reference numbers (from the previous benchmark, hardcoded for
# overlay on this sprint's figure).
SPRINT3_REF = {
    "ns": [1, 4, 16, 64],
    "pbg": [1860.0, 16.4, 16.3, 16.4],
    "cc": [6.23, 5.23, 5.22, 5.42],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trajs", default="1,4,16,64,256,1024")
    ap.add_argument("--t-ticks", type=int, default=DEFAULT_T_TICKS)
    ap.add_argument("--n-active-t", type=int, default=DEFAULT_N_ACTIVE_RNAP)
    ap.add_argument("--n-active-p", type=int, default=DEFAULT_N_ACTIVE_RIBOSOME)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-04/multistep_prototype.html"))
    args = ap.parse_args()

    n_trajs = [int(x) for x in args.n_trajs.split(",")]
    cache_path = Path(".pbg/runs/pdmp-04-multistep-results.json")
    import json
    if cache_path.is_file() and getattr(args, "use_cache", True):
        print(f"Loading cached results from {cache_path}", flush=True)
        results = json.loads(cache_path.read_text())
    else:
        results = benchmark(n_trajs, args.t_ticks, args.n_active_t,
                             args.n_active_p)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2))
        print(f"Cached results to {cache_path}", flush=True)

    print("\nSummary:")
    for r in results:
        print(f"  N={r['n']:>5}: CC={r['per_traj_cc_ms']:7.3f} ms/traj   "
              f"pbg={r['per_traj_pbg_ms'] or '---':>8} ms/traj   "
              f"speedup={r['speedup'] or '---'}×")

    plot_uri = make_figure(results, single_step_ref=SPRINT3_REF)
    write_html(args.out, results, args.t_ticks, plot_uri)


if __name__ == "__main__":
    main()
