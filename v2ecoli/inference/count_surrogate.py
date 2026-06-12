from __future__ import annotations
import numpy as np

def fit_negbinom_dispersion(mean: float, var: float) -> float:
    """Method-of-moments dispersion phi for NegBinom (var = mean + mean^2/phi).
    Returns +inf when var <= mean (Poisson limit)."""
    if var <= mean:
        return float("inf")
    return float(mean * mean / (var - mean))

def count_surrogate_sample(theta, calib, n_obs, rng):
    """Draw n_obs counts from NegBinom(mean=theta*mu0, dispersion=phi)."""
    mean = float(theta) * float(calib["mu0"])
    phi = float(calib["phi"])
    if not np.isfinite(phi):
        return rng.poisson(mean, size=n_obs).astype(float)
    p = phi / (phi + mean)           # numpy negative_binomial(n=phi, p)
    return rng.negative_binomial(phi, p, size=n_obs).astype(float)

def count_summary(counts) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    return np.array([c.mean(), c.std()])
