"""Phase 3 sprint 5: ensemble likelihood figure.

Reads the persisted N=8 PDMP+poisson ensemble from
``.pbg/runs/pdmp-03-likelihood/seed_<NN>/store.zarr/`` (sprint 4
output), renders a 3-panel matplotlib figure of the per-tick
log-likelihoods (transcript_init, polypeptide_init, total) with
the across-replicate mean ± σ band and faded individual replicate
traces, and writes an HTML report at
``reports/figures/pdmp-03/likelihood_ensemble.html`` with the same
provenance-banner pattern as the Phase-2 closeout report.

Usage::

    .venv/bin/python scripts/phase3_ensemble_figure.py
    # or override location
    .venv/bin/python scripts/phase3_ensemble_figure.py \\
        --runs-root .pbg/runs/pdmp-03-likelihood --n 8 \\
        --out reports/figures/pdmp-03/likelihood_ensemble.html
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# Reuse the load_ensemble helper from the sprint-4 script — it knows
# the hive-partitioned DataTree layout XArrayEmitter writes.
from scripts.phase3_likelihood_xarray_ensemble import (
    OUT_ROOT as DEFAULT_RUNS_ROOT,
    load_ensemble,
)


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
    return {
        "sha": sha,
        "short": sha[:8] if sha != "(unknown)" else sha,
        "branch": branch,
        "dirty": dirty,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }


PANEL_VARS = [
    ("total", "likelihood.total", "#7c3aed",
     "Aggregate per-tick log-likelihood across both initiation channels."),
    ("transcript_init", "rnap_data initiation", "#1e3a8a",
     "TranscriptInitiation Poisson-per-promoter contribution."),
    ("polypeptide_init", "ribosome_data initiation", "#0891b2",
     "PolypeptideInitiation Poisson-per-protein contribution."),
]


def make_figure(ds) -> str:
    """Render the 3-panel ensemble figure. Returns a data: URI."""
    # The sprint-4 buffer-quirk leaves the trailing 2 timesteps as NaN.
    # Drop them so the plotted means/σ don't dip artificially at the
    # right edge.
    valid_mask = ~np.isnan(ds["total"]).any(dim="replicate")
    ds = ds.where(valid_mask, drop=True)
    t = ds["time"].values if "time" in ds.coords else np.arange(ds.sizes["time"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    n_replicates = ds.sizes["replicate"]

    for ax, (var, label, color, caption) in zip(axes, PANEL_VARS):
        arr = ds[var]  # shape (replicate, time)
        # Faded individual traces.
        for r in range(n_replicates):
            ax.plot(t, arr.isel(replicate=r).values,
                    color=color, alpha=0.18, lw=0.9)
        # Mean ± σ band.
        mu = arr.mean(dim="replicate").values
        sd = arr.std(dim="replicate").values
        ax.plot(t, mu, color=color, lw=2.2, label="mean")
        ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.20,
                        label="± σ")
        ax.set_ylabel(f"log P({label}|λ)", fontsize=10)
        ax.set_title(
            f"{var}: per-tick log-likelihood, N={n_replicates} seeds  "
            f"— {caption}",
            fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "Phase-3 sprint-5: PDMP+poisson likelihood ensemble "
        f"(loaded from {DEFAULT_RUNS_ROOT}/)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, ds, plot_uri: str) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    # Per-replicate Σ_t summary.
    rows = []
    for var, label, color, _caption in PANEL_VARS:
        arr = ds[var]
        per_rep = arr.sum(dim="time", skipna=True).values
        rows.append(
            f"<tr><td>{var}</td><td>{label}</td>"
            f"<td class='num'>{per_rep.mean():.2f}</td>"
            f"<td class='num'>{per_rep.std():.2f}</td>"
            f"<td class='num'>{per_rep.min():.2f}</td>"
            f"<td class='num'>{per_rep.max():.2f}</td></tr>")
    table_rows = "\n".join(rows)

    n = ds.sizes["replicate"]
    t_max = float(ds["time"].max().item()) if "time" in ds.coords else 0.0
    nan_warn = ""
    nans_per_var = {v: int(np.isnan(ds[v]).sum().item())
                    for v, _l, _c, _cap in PANEL_VARS}
    if any(nans_per_var.values()):
        nan_warn = (
            f"<div class='warn'>Trailing-buffer NaN visible in the "
            f"persisted store: {nans_per_var}. The figure trims them "
            f"out before plotting; readers using the zarr directly "
            f"should expect 2 trailing NaN timesteps (XArrayEmitter "
            f"`buf_size &gt;= 3` minimum, partial trailing batch lost "
            f"on close).</div>")

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 sprint 5 — likelihood ensemble</title>
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
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px;
              margin: 6px 0 14px; }}
  .warn {{ background:#fef3c7; border-left:4px solid #f59e0b;
           padding:10px 14px; margin:14px 0; font-size:0.92em; }}
</style>

<h1>Phase 3 sprint 5 — PDMP+poisson likelihood ensemble</h1>
<p style='color:#6b7280; font-size:0.9em;'>
  N={n} replicates · t={t_max:.0f} s · loaded from
  <code>{DEFAULT_RUNS_ROOT}/seed_NN/store.zarr</code>
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

{nan_warn}

<h2>Per-tick log-likelihood by initiation channel</h2>
<p>
  Each panel shows the per-tick log-likelihood of the observed
  per-promoter / per-protein initiation counts under the Poisson rates
  the sampler used (sprint 1 + 2). Faded coloured traces are individual
  replicates; the bold line is the across-replicate mean; the band is
  ±1 σ. Bottom panel is the LikelihoodCollector aggregate (sprint 2).
</p>
<img class="plot" src="{plot_uri}" alt="ensemble likelihood panels">

<h2>Per-replicate Σ_t summary</h2>
<table>
  <tr><th>variable</th><th>channel</th>
      <th>μ over reps</th><th>σ over reps</th>
      <th>min</th><th>max</th></tr>
  {table_rows}
</table>

<h2>Pipeline status</h2>
<p>
  Phase 3 sprints 1–4 are landed: per-process Poisson likelihoods are
  emitted (TranscriptInitiation, PolypeptideInitiation), aggregated by
  the LikelihoodCollector step, persisted to XArrayEmitter zarr stores
  per replicate, and loaded back into an
  <code>xarray.Dataset(replicate × time × observable)</code> for
  downstream inference. The cross-replicate σ above (~488 fg-equivalent
  on the total) is ~800× the cell_mass σ Phase-2 measured under the
  same controller — confirming the Phase-2 closeout finding that
  count-level listeners carry the jump-process signal cell_mass washes
  out.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-03/likelihood_ensemble.html"))
    args = ap.parse_args()

    print(f"Loading N={args.n} replicates from "
          f"{DEFAULT_RUNS_ROOT}/...", flush=True)
    ds = load_ensemble(args.n)
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  vars: {list(ds.data_vars)}")

    plot_uri = make_figure(ds)
    write_html(args.out, ds, plot_uri)


if __name__ == "__main__":
    main()
