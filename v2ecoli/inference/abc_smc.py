from __future__ import annotations
import numpy as np


def abc_smc(observed_summary, simulate, prior, *, n_particles=400,
            n_generations=5, eps_quantile=0.5, rng):
    """Sequential Monte Carlo ABC over a scalar theta.

    simulate(theta, rng) -> summary vector (same shape as observed_summary).
    prior = (lo, hi) uniform. Returns particles, normalized weights, and
    eps/ess/accept traces. Distance = Euclidean on summaries.
    """
    lo, hi = float(prior[0]), float(prior[1])
    obs = np.asarray(observed_summary, dtype=float)

    def dist(theta, r):
        return float(np.linalg.norm(np.asarray(simulate(theta, r), float) - obs))

    parts = rng.uniform(lo, hi, size=n_particles)
    dists = np.array([dist(t, rng) for t in parts])
    weights = np.full(n_particles, 1.0 / n_particles)
    eps_trace, ess_trace, accept_trace = [float(dists.max())], [], []

    for _ in range(1, n_generations):
        eps = float(np.quantile(dists, eps_quantile))
        mean_w = np.average(parts, weights=weights)
        var_w = np.average((parts - mean_w) ** 2, weights=weights)
        kstd = max(np.sqrt(2.0 * var_w), 1e-6)
        new_parts = np.empty(n_particles)
        new_w = np.empty(n_particles)
        new_d = np.empty(n_particles)
        tries = 0
        i = 0
        while i < n_particles:
            tries += 1
            j = rng.choice(n_particles, p=weights)
            cand = parts[j] + rng.normal(0.0, kstd)
            if cand < lo or cand > hi:
                continue
            d = dist(cand, rng)
            if d <= eps:
                kern = np.exp(-0.5 * ((cand - parts) / kstd) ** 2)
                denom = np.sum(weights * kern)
                new_parts[i] = cand
                new_w[i] = (1.0 / (hi - lo)) / denom if denom > 0 else 0.0
                new_d[i] = d
                i += 1
        parts, dists = new_parts, new_d
        s = new_w.sum()
        weights = new_w / s if s > 0 else np.full(n_particles, 1.0 / n_particles)
        eps_trace.append(eps)
        ess_trace.append(float(1.0 / np.sum(weights ** 2)))
        accept_trace.append(float(n_particles / max(tries, 1)))

    return {"particles": parts, "weights": weights, "eps_trace": eps_trace,
            "ess_trace": ess_trace, "accept_trace": accept_trace}
