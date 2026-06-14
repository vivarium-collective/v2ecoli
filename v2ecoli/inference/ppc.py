from __future__ import annotations
import numpy as np
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary

def ppc_coverage(posterior, calib, *, n_obs, n_heldout, theta_true, rng):
    """Fraction of held-out observed summaries whose [mean] component falls in
    the posterior-predictive 90% interval."""
    pp = np.array([count_summary(count_surrogate_sample(
        rng.choice(posterior), calib, n_obs, rng))[0] for _ in range(400)])
    lo, hi = np.quantile(pp, [0.05, 0.95])
    held = np.array([count_summary(count_surrogate_sample(
        theta_true, calib, n_obs, rng))[0] for _ in range(n_heldout)])
    return float(np.mean((held >= lo) & (held <= hi)))
