import numpy as np
from v2ecoli.inference.ppc import ppc_coverage

def test_ppc_coverage_in_range():
    rng = np.random.default_rng(0)
    calib = {"mu0": 40.0, "phi": 8.0}
    posterior = rng.normal(1.0, 0.03, size=300)
    cov = ppc_coverage(posterior, calib, n_obs=300, n_heldout=200,
                       theta_true=1.0, rng=rng)
    assert 0.75 <= cov <= 1.0
