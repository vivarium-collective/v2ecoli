from types import SimpleNamespace

import numpy as np

from scripts._compare.parca_section import final_sim_data_diff


def _fake_sim_data(dry_mass, expression):
    # mirrors the attr paths used by parca_compare.SCALARS / DISTRIBUTIONS
    return SimpleNamespace(
        mass=SimpleNamespace(
            avg_cell_dry_mass=dry_mass,
            avg_cell_dry_mass_init=dry_mass,
            avg_cell_water_mass_init=0.0,
            fitAvgSolubleTargetMolMass=0.0,
        ),
        constants=SimpleNamespace(darkATP=1.0),
        process=SimpleNamespace(
            transcription=SimpleNamespace(
                rna_expression={"basal": expression},
            ),
        ),
    )


def test_final_sim_data_diff_flags_matching_and_drifting():
    left = _fake_sim_data(2.5, np.array([0.1, 0.2, 0.3]))
    right = _fake_sim_data(2.5, np.array([0.1, 0.2, 0.3]) * 1.01)
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    by_label = {r["label"]: r for r in rows}
    assert by_label["mass.avg_cell_dry_mass"]["verdict"] == "within_tol"
    assert by_label["RNA expression — basal"]["verdict"] == "drift"


def test_final_sim_data_diff_missing_attr_is_not_compared():
    left = _fake_sim_data(2.5, np.array([0.1]))
    right = SimpleNamespace()  # nothing reachable
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    assert all(r["verdict"] == "not_compared" for r in rows)
