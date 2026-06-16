"""Tests for the Shape step (capsule geometry from mass)."""
import math

from v2ecoli.structural.shape import capsule_from_mass


def test_width_and_density_fixed_length_grows_with_mass():
    base = capsule_from_mass(1200.0, width_um=1.0, density_g_per_ml=1.1)
    big = capsule_from_mass(2400.0, width_um=1.0, density_g_per_ml=1.1)
    # width + density are fixed; volume scales with mass; length grows.
    assert base["width_um"] == big["width_um"] == 1.0
    assert math.isclose(big["volume_fl"], 2 * base["volume_fl"], rel_tol=1e-9)
    assert big["length_um"] > base["length_um"]


def test_volume_is_mass_over_density():
    # 1100 fg at 1.1 g/mL (= 1100 fg/fL) -> 1.0 fL.
    s = capsule_from_mass(1100.0, density_g_per_ml=1.1)
    assert math.isclose(s["volume_fl"], 1.0, rel_tol=1e-9)


def test_capsule_volume_matches_derived_volume():
    # The capsule's geometric volume (pi*r^2*L_cyl + 4/3*pi*r^3) == mass/density.
    s = capsule_from_mass(2500.0, width_um=1.0, density_g_per_ml=1.1)
    r, hl = s["radius_A"], s["half_len_A"]
    lcyl = 2 * hl  # cylinder length = 2 * half_len
    v_A3 = math.pi * r ** 2 * lcyl + (4.0 / 3.0) * math.pi * r ** 3
    assert math.isclose(v_A3 / 1e12, s["volume_fl"], rel_tol=1e-6)


def test_length_includes_caps():
    # tip-to-tip length = cylinder (2*half_len) + 2 caps (2*r).
    s = capsule_from_mass(2500.0, width_um=1.0, density_g_per_ml=1.1)
    expected_A = 2 * s["half_len_A"] + 2 * s["radius_A"]
    assert math.isclose(s["length_um"], expected_A / 1e4, rel_tol=1e-9)
