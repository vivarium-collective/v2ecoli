import numpy as np
from v2ecoli.inference.abc_smc import abc_smc


def test_abc_smc_recovers_2d_mean():
    rng = np.random.default_rng(0)
    truth = np.array([2.0, -1.0])

    def simulate(theta, r):
        th = np.atleast_1d(theta)
        return r.normal(th, 0.5, size=(40, 2)).mean(axis=0)   # 2-D summary

    observed = simulate(truth, np.random.default_rng(99))
    res = abc_smc(observed, simulate, prior=[(0.0, 4.0), (-3.0, 1.0)],
                  n_particles=600, n_generations=7, rng=rng)
    parts = np.asarray(res["particles"])   # shape (n_particles, 2)
    assert parts.shape == (600, 2)
    post_mean = np.average(parts, axis=0, weights=res["weights"])
    assert np.linalg.norm(post_mean - truth) < 0.4          # recovers 2-D truth
    assert res["eps_trace"][-1] < res["eps_trace"][0]
    assert abs(res["weights"].sum() - 1.0) < 1e-9
