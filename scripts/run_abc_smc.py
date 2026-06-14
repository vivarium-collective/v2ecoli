"""Headline Phase-3 posterior recovery on the calibrated count surrogate.

Loads the real WCM count-surrogate calibration
(`.pbg/runs/phase3-surrogate-calibration.json`: mu0=12593.5, phi=inf ->
Poisson), recovers a scalar theta on synthetic data via ABC-SMC, prints the
posterior mean + 90% CI vs. the ground-truth theta, and writes a
self-contained HTML figure (weighted posterior histogram, base64 PNG
data-uri) to `reports/figures/pdmp-03/abc_smc_posterior.html`.

Usage:
    .venv/bin/python scripts/run_abc_smc.py
"""
from __future__ import annotations
import base64
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v2ecoli.inference.recover import recover

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = REPO_ROOT / ".pbg/runs/phase3-surrogate-calibration.json"
VIZ_OUT = REPO_ROOT / "reports/figures/pdmp-03/abc_smc_posterior.html"


def fig_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_fig(res):
    p = np.asarray(res["particles"], dtype=float)
    w = np.asarray(res["weights"], dtype=float)
    theta_true = res["theta_true"]
    lo, hi = res["ci90"]
    post_mean = res["post_mean"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p, bins=40, weights=w, density=True, color="#1f77b4",
            alpha=0.6, label="ABC-SMC posterior")
    ax.axvspan(lo, hi, color="#1f77b4", alpha=0.12,
               label=f"90% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(theta_true, color="#d62728", lw=2.0,
               label=f"theta_true = {theta_true:.3f}")
    ax.axvline(post_mean, color="black", lw=1.5, ls="--",
               label=f"post_mean = {post_mean:.3f}")
    ax.set_xlabel("theta")
    ax.set_ylabel("weighted posterior density")
    ax.set_title("Phase-3 ABC-SMC posterior recovery (calibrated surrogate)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def write_html(out_path: Path, fig_data_uri: str, res: dict, calib: dict):
    theta_true = res["theta_true"]
    lo, hi = res["ci90"]
    post_mean = res["post_mean"]
    contains = "yes" if lo <= theta_true <= hi else "no"
    phi_str = "inf (Poisson)" if not np.isfinite(calib["phi"]) else f"{calib['phi']:.3g}"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Phase-3 ABC-SMC posterior recovery</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 22px; }}
  table.cmp {{ border-collapse: collapse; margin: 14px 0; }}
  table.cmp th, table.cmp td {{ padding: 7px 14px; border: 1px solid #ccc; }}
  table.cmp th {{ background: rgba(0,0,0,0.03); text-align: left; }}
  table.cmp td.num {{ text-align: right; font-variant-numeric: tabular-nums;
                      font-family: ui-monospace, Menlo, monospace; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
</style></head>
<body>
<header style="padding: 18px 22px; border-bottom: 1px solid #eee;">
  <h1>Phase-3 ABC-SMC posterior recovery</h1>
  <p>Calibrated count surrogate (mu0={calib["mu0"]:.1f}, phi={phi_str}) ·
     auto-generated {time.strftime("%Y-%m-%d")}</p>
</header>
<main>
<p>
ABC-SMC recovers a scalar abundance-scaling theta on synthetic count data
drawn from the calibrated WCM count surrogate. The histogram below is the
weighted posterior over particles; the ground-truth theta and the weighted
90% credible interval are marked.
</p>
<table class="cmp">
  <tr><th>quantity</th><th>value</th></tr>
  <tr><td>theta_true</td><td class="num">{theta_true:.4f}</td></tr>
  <tr><td>post_mean</td><td class="num">{post_mean:.4f}</td></tr>
  <tr><td>90% CI</td><td class="num">[{lo:.4f}, {hi:.4f}]</td></tr>
  <tr><td>CI width</td><td class="num">{hi - lo:.4f}</td></tr>
  <tr><td>CI contains truth</td><td class="num">{contains}</td></tr>
</table>
<img src="{fig_data_uri}" alt="ABC-SMC weighted posterior histogram">
</main>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    calib = json.loads(CALIB_PATH.read_text())
    calib = {"mu0": float(calib["mu0"]), "phi": float(calib["phi"])}

    res = recover(calib, theta_true=1.2, n_obs=400,
                  rng=np.random.default_rng(0))

    lo, hi = res["ci90"]
    print(f"theta_true = {res['theta_true']:.4f}")
    print(f"post_mean  = {res['post_mean']:.4f}")
    print(f"ci90       = [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
    print(f"CI contains truth: {lo <= res['theta_true'] <= hi}")

    fig = make_fig(res)
    data_uri = fig_to_data_uri(fig)
    VIZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_html(VIZ_OUT, data_uri, res, calib)
    print(f"\nWrote {VIZ_OUT}")


if __name__ == "__main__":
    main()
