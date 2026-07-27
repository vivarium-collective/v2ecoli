"""Fast unit tests for the deep sim_data injection adapter routing (no sims)."""
import numpy as np

from pbg_v2ecoli.uq_sim_data_injection import (
    POST_PARCA, REBUILD, DeepParam,
    rnap_elongation_rate, cell_dry_mass_fraction, kinetic_objective_weight,
)


def test_param_modes_and_bounds():
    ps = [rnap_elongation_rate(), cell_dry_mass_fraction(), kinetic_objective_weight()]
    modes = {p.name: p.mode for p in ps}
    assert modes["rnap_elongation_rate"] == POST_PARCA
    assert modes["cell_dry_mass_fraction"] == REBUILD
    assert modes["kinetic_objective_weight"] == POST_PARCA
    b = np.array([list(p.bounds) for p in ps])
    assert b.shape == (3, 2)
    assert (b[:, 0] < b[:, 1]).all()


def test_post_parca_override_keys():
    # kinetic weight -> a single float config key on ecoli-metabolism
    kow = kinetic_objective_weight()
    ov = kow.make_overrides(3e-7, {})
    assert ov == {"ecoli-metabolism.kinetic_objective_weight": 3e-7}

    # rnap elongation -> scales the per-condition dict on ecoli-transcript-elongation
    rer = rnap_elongation_rate()
    configs = {"ecoli-transcript-elongation":
               {"rnaPolymeraseElongationRateDict": {"minimal": 50.0, "rich": 60.0}}}
    ov = rer.make_overrides(1.2, configs)
    key = "ecoli-transcript-elongation.rnaPolymeraseElongationRateDict"
    assert set(ov) == {key}
    assert ov[key] == {"minimal": 60.0, "rich": 72.0}
    # original untouched (deepcopy)
    assert configs["ecoli-transcript-elongation"]["rnaPolymeraseElongationRateDict"]["minimal"] == 50.0


def test_rebuild_mutate_sets_attr():
    dmf = cell_dry_mass_fraction()

    class _Mass:
        cell_dry_mass_fraction = 0.30

    class _SD:
        mass = _Mass()

    sd = _SD()
    dmf.mutate(0.27, sd)
    assert sd.mass.cell_dry_mass_fraction == 0.27


def test_bad_param_config_raises():
    import pytest
    with pytest.raises(ValueError):
        DeepParam("x", "a.b", POST_PARCA, (0, 1))  # missing make_overrides
    with pytest.raises(ValueError):
        DeepParam("x", "a.b", REBUILD, (0, 1))  # missing mutate
