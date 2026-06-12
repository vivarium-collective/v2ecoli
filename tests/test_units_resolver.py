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
