from __future__ import annotations
import numpy as np
from scipy import stats
from v2ecoli.inference.abc_smc import abc_smc


def posterior_rank(theta_star, particles, weights) -> int:
    """Effective rank of theta_star in the (weighted) posterior.

    ABC-SMC returns *weighted* importance particles, so the SBC rank is the
    weighted count below theta_star, i.e. the weight mass strictly below
    theta_star scaled to [0, n_particles]. For equal weights this reduces to
    the plain count of particles strictly below theta_star.
    """
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    w = weights / weights.sum()
    frac_below = float(np.sum(w[particles < float(theta_star)]))
    return int(round(len(particles) * frac_below))


def run_sbc(simulate, prior, *, n_sbc=150, abc_kwargs, rng):
    """For n_sbc draws: theta* ~ prior, simulate observed, ABC-SMC posterior,
    record rank(theta*). Return ranks, chi2 p-value of rank-uniformity, hist."""
    lo, hi = float(prior[0]), float(prior[1])
    ranks = []
    for _ in range(n_sbc):
        theta_star = rng.uniform(lo, hi)
        observed = simulate(theta_star, rng)
        res = abc_smc(observed, simulate, prior=prior, rng=rng, **abc_kwargs)
        ranks.append(posterior_rank(theta_star, res["particles"], res["weights"]))
    ranks = np.array(ranks)
    n_part = abc_kwargs["n_particles"]
    n_bins = 20
    hist, edges = np.histogram(ranks, bins=n_bins, range=(0, n_part))
    expected = len(ranks) / n_bins
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    chi2_p = float(stats.chi2.sf(chi2, df=n_bins - 1))
    return {"ranks": ranks, "chi2_p": chi2_p, "hist": hist, "edges": edges}
