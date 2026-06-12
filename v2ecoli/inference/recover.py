from __future__ import annotations
import numpy as np
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary
from v2ecoli.inference.abc_smc import abc_smc


def recover(calib, theta_true, n_obs, rng, *, prior=(0.2, 3.0),
            n_particles=400, n_generations=6):
    """Recover a scalar theta on synthetic count data via ABC-SMC.

    Draws an observed summary from the calibrated count surrogate at
    ``theta_true``, runs ABC-SMC against the same simulator, and returns the
    posterior particles plus a weighted 90% credible interval (``ci90``).
    """
    def simulate(theta, r):
        return count_summary(count_surrogate_sample(theta, calib, n_obs, r))

    observed = simulate(theta_true, np.random.default_rng(12345))
    res = abc_smc(observed, simulate, prior=prior, rng=rng,
                  n_particles=n_particles, n_generations=n_generations)
    p, w = res["particles"], res["weights"]
    order = np.argsort(p)
    cw = np.cumsum(w[order])
    lo = float(p[order][np.searchsorted(cw, 0.05)])
    hi = float(p[order][np.searchsorted(cw, 0.95)])
    return {**res, "theta_true": theta_true,
            "post_mean": float(np.average(p, weights=w)), "ci90": (lo, hi)}
