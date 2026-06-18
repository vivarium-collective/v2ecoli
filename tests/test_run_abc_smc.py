import numpy as np
from v2ecoli.inference.recover import recover


def test_recovery_ci_contains_truth():
    calib = {"mu0": 40.0, "phi": 8.0}
    res = recover(calib, theta_true=1.2, n_obs=400,
                  rng=np.random.default_rng(0),
                  n_particles=400, n_generations=6)
    lo, hi = res["ci90"]
    assert lo <= 1.2 <= hi                       # 90% CI contains truth
    assert (hi - lo) < 0.6                        # informative posterior
