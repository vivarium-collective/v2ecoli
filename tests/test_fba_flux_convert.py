from v2ecoli.library.fba_flux_convert import millard_flux_to_fba_bound


def test_converter_is_linear_and_scaled():
    # bound = flux_mM_per_s * coefficient * scale; pure, deterministic
    assert millard_flux_to_fba_bound(2.0, coefficient=3.0, scale=1.0) == 6.0
    assert millard_flux_to_fba_bound(2.0, coefficient=3.0, scale=0.5) == 3.0


def test_converter_sign_preserved():
    assert millard_flux_to_fba_bound(-1.5, coefficient=2.0, scale=1.0) == -3.0
