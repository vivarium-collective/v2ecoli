"""Phase-3 increment 2: 2-parameter (mean-scale, dispersion-scale) joint
inference on count data, demonstrating the N-D ABC-SMC engine + joint
identifiability, validated by per-parameter SBC.

The forward model is NegBinom(mean = th1*mu0, dispersion = th2*phi0). The
[mean, std] summary carries both the mean (-> th1) and the variance (-> th2),
so the two parameters are jointly identifiable. SBC validates calibration for
EACH parameter's marginal.

NOTE ON GROUNDING: this uses a SYNTHETIC overdispersed calibration (mu0=40,
phi0=8). The real WCM transcription count (rna_init_event) is Poisson
(phi=inf, no dispersion to infer -- see .pbg/runs/phase3-surrogate-calibration.json),
so this validates the multi-parameter machinery for when richer
overdispersed / multi-modal WCM observables are added (a later increment).

Usage: .venv/bin/python scripts/run_abc_smc_2d.py [--n-sbc 80]
Writes reports/figures/pdmp-03/abc_smc_2d.html
"""
from __future__ import annotations
import argparse, base64, io, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.inference.abc_smc import abc_smc

MU0, PHI0 = 40.0, 8.0
PRIOR = [(0.5, 2.0), (0.3, 3.0)]
TRUTH = np.array([1.2, 1.0])


def _nb(mean, phi, n, rng):
    p = phi / (phi + mean)
    return rng.negative_binomial(phi, p, size=n).astype(float)


def simulate(theta, rng):
    th = np.atleast_1d(theta)
    return np.array(_summary(_nb(th[0] * MU0, max(th[1] * PHI0, 0.2), 600, rng)))


def _summary(c):
    return [float(np.mean(c)), float(np.std(c))]


def _sbc_2d(n_sbc, abc_kwargs, rng):
    ranks = [[], []]
    for _ in range(n_sbc):
        ts = np.array([rng.uniform(l, h) for l, h in PRIOR])
        obs = simulate(ts, rng)
        r = abc_smc(obs, simulate, prior=PRIOR, rng=rng, **abc_kwargs)
        P = np.asarray(r["particles"])
        for d in range(2):
            ranks[d].append(int(np.sum(P[:, d] < ts[d])))
    out = {}
    for d in range(2):
        h, _ = np.histogram(ranks[d], bins=20, range=(0, abc_kwargs["n_particles"]))
        e = n_sbc / 20.0
        out[d] = float(stats.chi2.sf(float(np.sum((h - e) ** 2 / e)), df=19))
    return out


def main(n_sbc):
    observed = simulate(TRUTH, np.random.default_rng(99))
    res = abc_smc(observed, simulate, prior=PRIOR, n_particles=600,
                  n_generations=10, rng=np.random.default_rng(0))
    parts, w = np.asarray(res["particles"]), res["weights"]
    pm = np.average(parts, axis=0, weights=w)
    cov = np.cov(parts.T, aweights=w)
    corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    p = _sbc_2d(n_sbc, dict(n_particles=300, n_generations=10), np.random.default_rng(1))
    print(f"2-param recovery: post_mean=[{pm[0]:.3f},{pm[1]:.3f}] (truth [1.2,1.0]) "
          f"corr={corr:+.3f}")
    print(f"SBC chi2_p: mean-scale={p[0]:.3f} {'PASS' if p[0]>0.05 else 'FAIL'} | "
          f"disp-scale={p[1]:.3f} {'PASS' if p[1]>0.05 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(parts[:, 0], parts[:, 1], s=8, alpha=0.3, c="#1f77b4")
    ax.axvline(TRUTH[0], color="r", ls="--", lw=1)
    ax.axhline(TRUTH[1], color="r", ls="--", lw=1)
    ax.set_xlabel("mean-scale (theta1)"); ax.set_ylabel("dispersion-scale (theta2)")
    ax.set_title(f"2-param joint posterior  corr={corr:+.2f}  "
                 f"SBC p=({p[0]:.2f},{p[1]:.2f})")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    out = Path("reports/figures/pdmp-03/abc_smc_2d.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!DOCTYPE html><html><body><h1>Phase-3 increment 2: 2-parameter joint "
        "inference + identifiability</h1><p>NegBinom(mean=th1*40, disp=th2*8); "
        f"recovery post_mean=[{pm[0]:.3f},{pm[1]:.3f}] (truth [1.2,1.0]); posterior "
        f"correlation={corr:+.3f} (weak -> jointly identifiable); SBC chi2_p "
        f"mean-scale={p[0]:.3f}, disp-scale={p[1]:.3f} (both PASS if &gt;0.05). "
        "Synthetic overdispersion (the WCM transcription count is Poisson).</p>"
        f"<img src='{uri}'></body></html>")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sbc", type=int, default=80)
    a = ap.parse_args(); main(a.n_sbc)
