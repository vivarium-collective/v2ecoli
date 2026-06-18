"""Phase-3 SBC gate: rank-histogram uniformity on the WCM-calibrated count
surrogate. Loads .pbg/runs/phase3-surrogate-calibration.json, runs SBC, writes
reports/figures/pdmp-03/sbc_rank_histogram.html with chi2 p + PASS/FAIL.
Usage: .venv/bin/python scripts/run_sbc.py [--n-sbc 150]
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary
from v2ecoli.inference.sbc import run_sbc

def main(n_sbc):
    calib = json.loads(Path(".pbg/runs/phase3-surrogate-calibration.json").read_text())
    calib = {"mu0": calib["mu0"], "phi": float(calib["phi"])}  # phi may be "Infinity"
    def simulate(theta, r):
        return count_summary(count_surrogate_sample(theta, calib, 400, r))
    # n_generations=12 (not 6): the WCM-calibrated count likelihood is FAR
    # sharper than the test surrogate -- mu0=12593 with n_obs=400 gives a true
    # theta posterior std ~ sqrt(mu0/n_obs)/mu0 ~ 0.0004, vs prior width 2.8.
    # ABC-SMC halves eps per generation, so 5 steps (ng=6) leave the posterior
    # ~40x too wide -> overdispersed -> SBC *correctly* rejects (chi2_p=0). That
    # is ABC underconvergence, not a calibration defect; giving ABC enough
    # generations to converge is the legitimate precondition for the uniformity
    # gate. ng=9 -> sd ~5.6x too wide (still fails); ng=12 -> converged (uniform).
    res = run_sbc(simulate, prior=(0.2, 3.0), n_sbc=n_sbc,
                  abc_kwargs=dict(n_particles=300, n_generations=12),
                  rng=np.random.default_rng(0))
    passed = res["chi2_p"] > 0.05
    print(f"SBC: n_sbc={n_sbc} chi2_p={res['chi2_p']:.4f} -> {'PASS' if passed else 'FAIL'}")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(res["hist"])), res["hist"], width=1.0, color="#1f77b4", alpha=0.8)
    exp = len(res["ranks"])/len(res["hist"])
    ax.axhline(exp, color="black", ls="--", lw=1, label=f"uniform expected={exp:.1f}")
    ax.set_xlabel("rank bin"); ax.set_ylabel("count")
    ax.set_title(f"SBC rank histogram  chi2 p={res['chi2_p']:.3f}  {'PASS' if passed else 'FAIL'}")
    ax.legend(fontsize=8)
    buf=io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110); plt.close(fig)
    uri="data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    out=Path("reports/figures/pdmp-03/sbc_rank_histogram.html"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!DOCTYPE html><html><body><h1>Phase-3 SBC rank-histogram uniformity</h1>"
                   f"<p>n_sbc={n_sbc}, chi2 p={res['chi2_p']:.4f} -> <b>{'PASS' if passed else 'FAIL'}</b> "
                   f"(uniform if p&gt;0.05). Surrogate: WCM-calibrated count likelihood "
                   f"(mu0={calib['mu0']:.1f}, {'Poisson' if not np.isfinite(calib['phi']) else 'NegBinom'}).</p>"
                   f"<img src='{uri}'></body></html>")
    print("wrote", out)

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n-sbc", type=int, default=150)
    a=ap.parse_args(); main(a.n_sbc)
