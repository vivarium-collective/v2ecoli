import numpy as np
from v2ecoli.inference.count_surrogate import (
    fit_negbinom_dispersion, count_surrogate_sample, count_summary)

def test_fit_dispersion_method_of_moments():
    # var = mean + mean^2/phi  =>  phi = mean^2 / (var - mean)
    assert abs(fit_negbinom_dispersion(mean=10.0, var=30.0) - 5.0) < 1e-9
    # var <= mean -> Poisson limit (inf dispersion)
    assert not np.isfinite(fit_negbinom_dispersion(mean=10.0, var=10.0))

def test_sample_moments_match_calibration():
    rng = np.random.default_rng(0)
    calib = {"mu0": 40.0, "phi": 8.0}
    x = count_surrogate_sample(theta=1.0, calib=calib, n_obs=200000, rng=rng)
    # mean ~ theta*mu0 = 40 ; var ~ mean + mean^2/phi = 40 + 1600/8 = 240
    assert abs(x.mean() - 40.0) < 1.0
    assert abs(x.var() - 240.0) < 12.0

def test_theta_scales_mean():
    rng = np.random.default_rng(1)
    calib = {"mu0": 40.0, "phi": 8.0}
    x = count_surrogate_sample(theta=1.5, calib=calib, n_obs=200000, rng=rng)
    assert abs(x.mean() - 60.0) < 1.5

def test_summary_shape():
    s = count_summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s.shape == (2,)            # [mean, std]
    assert abs(s[0] - 2.5) < 1e-9

def test_calibrate_from_counts():
    import numpy as np
    from v2ecoli.inference.count_surrogate import calibrate_from_counts
    rng = np.random.default_rng(0)
    samples = rng.negative_binomial(8, 8/(8+40), size=5000).astype(float)
    calib = calibrate_from_counts(samples)
    assert abs(calib["mu0"] - 40) < 2
    assert abs(calib["phi"] - 8) < 3
    assert calib["theta_ref"] == 1.0
