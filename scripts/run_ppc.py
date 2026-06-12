"""Phase-3 PPC gate: posterior-predictive coverage on the WCM-calibrated count
surrogate. Loads .pbg/runs/phase3-surrogate-calibration.json, runs a CONVERGED
recovery (n_generations=12) to obtain a posterior, then checks the fraction of
held-out observed means inside the posterior-predictive 90% band. Writes
reports/figures/pdmp-03/ppc_coverage.html with the band, held-out histogram,
coverage fraction, and PASS/FAIL (PASS if cov >= 0.80).
Usage: .venv/bin/python scripts/run_ppc.py
"""
from __future__ import annotations
import base64, io, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary
from v2ecoli.inference.recover import recover
from v2ecoli.inference.ppc import ppc_coverage

def main():
    calib = json.loads(Path(".pbg/runs/phase3-surrogate-calibration.json").read_text())
    calib = {"mu0": calib["mu0"], "phi": float(calib["phi"])}  # phi may be "Infinity"
    theta_true = 1.2
    n_obs = 400
    n_heldout = 300
    # ng=12: the WCM-calibrated Poisson posterior is sharp; fewer generations
    # leave ABC underconverged (see run_sbc.py rationale).
    res = recover(calib, theta_true=theta_true, n_obs=n_obs,
                  rng=np.random.default_rng(0), n_generations=12)
    cov = ppc_coverage(res["particles"], calib, n_obs=n_obs, n_heldout=n_heldout,
                       theta_true=theta_true, rng=np.random.default_rng(1))
    passed = cov >= 0.80
    print(f"PPC: cov={cov:.4f} -> {'PASS' if passed else 'FAIL'} (PASS if cov>=0.80)")

    # Recompute pp band + held-out for the figure (separate rng, illustrative).
    figrng = np.random.default_rng(2)
    pp = np.array([count_summary(count_surrogate_sample(
        figrng.choice(res["particles"]), calib, n_obs, figrng))[0] for _ in range(400)])
    lo, hi = np.quantile(pp, [0.05, 0.95])
    held = np.array([count_summary(count_surrogate_sample(
        theta_true, calib, n_obs, figrng))[0] for _ in range(n_heldout)])

    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(held, bins=30, color="#1f77b4", alpha=0.8, label="held-out observed mean")
    ax.axvspan(lo, hi, color="orange", alpha=0.25, label="posterior-predictive 90% band")
    ax.axvline(lo, color="orange", ls="--", lw=1); ax.axvline(hi, color="orange", ls="--", lw=1)
    ax.set_xlabel("count mean"); ax.set_ylabel("held-out replicates")
    ax.set_title(f"PPC coverage={cov:.3f}  {'PASS' if passed else 'FAIL'}")
    ax.legend(fontsize=8)
    buf=io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110); plt.close(fig)
    uri="data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    out=Path("reports/figures/pdmp-03/ppc_coverage.html"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!DOCTYPE html><html><body><h1>Phase-3 posterior-predictive coverage</h1>"
                   f"<p>theta_true={theta_true}, n_obs={n_obs}, n_heldout={n_heldout}, "
                   f"coverage={cov:.4f} -> <b>{'PASS' if passed else 'FAIL'}</b> "
                   f"(PASS if cov&gt;=0.80). Posterior from converged recovery (ng=12). "
                   f"Surrogate: WCM-calibrated count likelihood (mu0={calib['mu0']:.1f}, "
                   f"{'Poisson' if not np.isfinite(calib['phi']) else 'NegBinom'}).</p>"
                   f"<img src='{uri}'></body></html>")
    print("wrote", out)

if __name__ == "__main__":
    main()
