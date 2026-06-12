import pytest
from v2ecoli.core import build_core
from v2ecoli.library.units_resolver import unit_from_type

@pytest.fixture(scope="module")
def core():
    return build_core()

@pytest.mark.parametrize("type_str, expected", [
    ("quantity[float,fg]", "fg"),
    ("quantity[fg]", "fg"),
    ("quantity[array[float],mM]", "mM"),
    ("float[1/s]", "1/s"),
    ("integer[s]", "s"),
    ("array[float[mM]]", "mM"),
    ("overwrite[array[float[mM]]]", "mM"),     # wrapper unwrap
    ("overwrite[float[fg]]", "fg"),            # wrapper unwrap
    ("float", None),                            # empty units -> None
    ("string", None),                           # non-numeric -> None
    ("not_a_real_type_xyz", None),              # unresolvable -> None
])
def test_unit_from_type(core, type_str, expected):
    assert unit_from_type(type_str, core) == expected


from v2ecoli.library.units_resolver import units_from_schema

def test_units_from_schema_nested(core):
    schema = {
        "listeners": {
            "mass": {
                "cell_mass": {"_type": "quantity[float,fg]", "_default": 0},
                "dry_mass":  {"_type": "quantity[float,fg]", "_default": 0},
            },
            "fba_results": {
                "conc_updates": {"_type": "overwrite[array[float[mM]]]", "_default": []},
            },
        },
        "timestep": {"_type": "integer[s]", "_default": 1},
        "bulk": "bulk_array",          # no unit -> omitted
    }
    index = units_from_schema(schema, core)
    assert index["listeners.mass.cell_mass"] == "fg"
    assert index["listeners.mass.dry_mass"] == "fg"
    assert index["listeners.fba_results.conc_updates"] == "mM"
    assert index["timestep"] == "s"
    assert "bulk" not in index           # unitless leaves are omitted
