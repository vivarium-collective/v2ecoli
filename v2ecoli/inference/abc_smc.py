from __future__ import annotations
import numpy as np


def abc_smc(observed_summary, simulate, prior, *, n_particles=400,
            n_generations=5, eps_quantile=0.5, rng):
    """Sequential Monte Carlo ABC over a scalar OR vector theta.

    simulate(theta, rng) -> summary vector (same shape as observed_summary).
    prior is either a scalar ``(lo, hi)`` uniform tuple (1-D / scalar mode) or
    a list of ``(lo, hi)`` pairs, one per dimension (N-D mode). Returns
    particles, normalized weights, and eps/ess/accept traces. Distance =
    Euclidean on summaries.

    In scalar mode the returned ``particles`` is a 1-D array of shape
    ``(n_particles,)`` (back-compat). In N-D mode it is ``(n_particles, n_dim)``.
    The perturbation kernel is an independent Gaussian per dimension (bandwidth
    = sqrt(2 * weighted per-dim variance)); the importance weight uses the
    product of per-dim Gaussian kernels; the prior density is the product of
    1/(hi-lo) per dim; the in-bounds check is per dimension.
    """
    # Detect scalar mode: a 2-tuple of scalars vs a sequence of (lo, hi) pairs.
    scalar_mode = len(prior) == 2 and np.isscalar(prior[0])
    bounds = [(float(prior[0]), float(prior[1]))] if scalar_mode \
        else [(float(p[0]), float(p[1])) for p in prior]
    n_dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    obs = np.asarray(observed_summary, dtype=float)

    def theta_of(row):
        # Pass a bare scalar to simulate in scalar mode for exact back-compat.
        return float(row[0]) if scalar_mode else row

    def dist(row, r):
        return float(np.linalg.norm(
            np.asarray(simulate(theta_of(row), r), float) - obs))

    parts = rng.uniform(lo, hi, size=(n_particles, n_dim))
    dists = np.array([dist(t, rng) for t in parts])
    weights = np.full(n_particles, 1.0 / n_particles)
    eps_trace, ess_trace, accept_trace = [float(dists.max())], [], []

    prior_density = float(np.prod(1.0 / (hi - lo)))

    for _ in range(1, n_generations):
        eps = float(np.quantile(dists, eps_quantile))
        mean_w = np.average(parts, axis=0, weights=weights)
        var_w = np.average((parts - mean_w) ** 2, axis=0, weights=weights)
        kstd = np.maximum(np.sqrt(2.0 * var_w), 1e-6)  # per-dim bandwidth
        new_parts = np.empty((n_particles, n_dim))
        new_w = np.empty(n_particles)
        new_d = np.empty(n_particles)
        tries = 0
        i = 0
        while i < n_particles:
            tries += 1
            j = rng.choice(n_particles, p=weights)
            cand = parts[j] + rng.normal(0.0, kstd)
            if np.any(cand < lo) or np.any(cand > hi):
                continue
            d = dist(cand, rng)
            if d <= eps:
                # Product of per-dim Gaussian kernels over all source particles.
                kern = np.prod(
                    np.exp(-0.5 * ((cand - parts) / kstd) ** 2), axis=1)
                denom = np.sum(weights * kern)
                new_parts[i] = cand
                new_w[i] = prior_density / denom if denom > 0 else 0.0
                new_d[i] = d
                i += 1
        parts, dists = new_parts, new_d
        s = new_w.sum()
        weights = new_w / s if s > 0 else np.full(n_particles, 1.0 / n_particles)
        eps_trace.append(eps)
        ess_trace.append(float(1.0 / np.sum(weights ** 2)))
        accept_trace.append(float(n_particles / max(tries, 1)))

    out_parts = parts[:, 0] if scalar_mode else parts
    return {"particles": out_parts, "weights": weights, "eps_trace": eps_trace,
            "ess_trace": ess_trace, "accept_trace": accept_trace}
