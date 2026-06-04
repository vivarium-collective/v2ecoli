from types import SimpleNamespace

import numpy as np

from scripts._compare.parca_section import final_sim_data_diff


class _Unum:
    """Minimal Unum-like quantity exposing asNumber(), as ParCa values do."""
    def __init__(self, value):
        self._v = np.asarray(value, dtype=float)

    def asNumber(self, *a, **k):
        return self._v


def _vecoli_obj(dry_mass, expression):
    # vEcoli SimulationDataEcoli object form: .mass, .process.transcription
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


def _v2_dict(dry_mass, expression):
    # Real v2 checkpoint form: dict of subsystems, NO top-level 'process',
    # values may be Unum quantities.
    return {
        "mass": SimpleNamespace(
            avg_cell_dry_mass=_Unum(dry_mass),
            avg_cell_dry_mass_init=_Unum(dry_mass),
            avg_cell_water_mass_init=_Unum(0.0),
            fitAvgSolubleTargetMolMass=_Unum(0.0),
        ),
        "constants": SimpleNamespace(darkATP=1.0),
        "transcription": SimpleNamespace(
            rna_expression={"basal": expression},
        ),
    }


def test_vecoli_obj_vs_v2_dict_matching_and_drifting():
    # vEcoli object on the left, v2 dict-of-subsystems on the right.
    left = _vecoli_obj(2.5, np.array([0.1, 0.2, 0.3]))
    right = _v2_dict(2.5, np.array([0.1, 0.2, 0.3]) * 1.01)
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    by_label = {r["label"]: r for r in rows}
    # scalar matches across the two different container forms + Unum unwrap
    assert by_label["mass.avg_cell_dry_mass"]["verdict"] == "within_tol"
    # distribution: 1% drift, and the v2 'process' prefix was stripped
    assert by_label["RNA expression — basal"]["verdict"] == "drift"


def test_missing_attr_is_not_compared():
    left = _vecoli_obj(2.5, np.array([0.1]))
    right = SimpleNamespace()  # nothing reachable
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    assert all(r["verdict"] == "not_compared" for r in rows)
