import numpy as np
from v2ecoli.inference.abc_smc import abc_smc


def test_abc_smc_recovers_gaussian_mean():
    rng = np.random.default_rng(0)
    theta_true = 3.0

    def simulate(theta, r):
        return np.array([r.normal(theta, 1.0, size=50).mean()])

    observed = simulate(theta_true, np.random.default_rng(99))
    res = abc_smc(observed, simulate, prior=(0.0, 6.0),
                  n_particles=400, n_generations=5, rng=rng)
    post_mean = np.average(res["particles"], weights=res["weights"])
    assert abs(post_mean - theta_true) < 0.3
    assert res["eps_trace"][-1] < res["eps_trace"][0]
    assert np.all(np.isfinite(res["weights"])) and abs(res["weights"].sum() - 1.0) < 1e-9


def test_abc_smc_posterior_concentrates():
    rng = np.random.default_rng(1)

    def simulate(theta, r):
        return np.array([r.normal(theta, 1.0, size=50).mean()])

    observed = np.array([3.0])
    res = abc_smc(observed, simulate, prior=(0.0, 6.0),
                  n_particles=400, n_generations=6, rng=rng)
    var = np.average((res["particles"] - np.average(res["particles"], weights=res["weights"]))**2,
                     weights=res["weights"])
    assert var**0.5 < 0.5
