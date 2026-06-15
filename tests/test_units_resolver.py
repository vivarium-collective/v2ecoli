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


from v2ecoli.library.units_resolver import build_units_index

def test_build_units_index_covers_known_listeners():
    index = build_units_index()           # builds its own core; memoized
    # cell mass is declared quantity[float,fg] on multiple listener inputs
    assert index.get("listeners.mass.cell_mass") == "fg"
    # at least one concentration (mM) and one rate (1/s) somewhere
    units = set(index.values())
    assert "mM" in units
    assert any(u in ("1/s", "1 / second") for u in units)
    # index is non-trivial
    assert len(index) > 10

def test_build_units_index_is_memoized():
    a = build_units_index()
    b = build_units_index()
    assert a is b                          # same cached object


from v2ecoli.library.units_resolver import (
    resolve_unit, format_axis_label, V2EcoliUnitsResolver,
)

def test_resolve_unit_hit_miss():
    index = {"listeners.mass.cell_mass": "fg"}
    assert resolve_unit(index, "listeners.mass.cell_mass") == "fg"
    assert resolve_unit(index, "global_time") is None
    assert resolve_unit(index, "") is None
    # array element / sub-leaf path falls back to parent
    assert resolve_unit(index, "listeners.mass.cell_mass.3") == "fg"

def test_format_axis_label():
    assert format_axis_label("Mass", "fg") == "Mass (fg)"
    assert format_axis_label("Mass", None) == "Mass"
    assert format_axis_label("Mass (fg)", "fg") == "Mass (fg)"   # idempotent
    assert format_axis_label("", "fg") == "(fg)"

def test_resolver_is_callable():
    r = V2EcoliUnitsResolver()
    assert r("listeners.mass.cell_mass") == "fg"     # delegates to build_units_index
    assert r("nonexistent.path") is None


# --- resilient figure wrappers -------------------------------------------

def test_units_figure_to_html_fallback_when_base_lacks_hook(monkeypatch):
    """When the installed base has no figure_to_html, the wrapper still works."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import v2ecoli.library.units_resolver as ur

    class _StaleBase:           # a base that predates the units hook
        pass

    monkeypatch.setattr(ur, "_base_visualization", lambda: _StaleBase)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel("Mass")
    html = ur.units_figure_to_html(fig, [(ax, "y", "listeners.mass.cell_mass")])
    assert html.startswith('<img src="data:image/png;base64,')
    assert html.count("<img") == 1


def test_units_finalize_figure_fallback_labels_axis(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import v2ecoli.library.units_resolver as ur

    monkeypatch.setattr(ur, "_base_visualization", lambda: None)
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    ur.units_finalize_figure(fig, [(ax, "y", "listeners.mass.cell_mass")])
    assert ax.get_ylabel() == "Mass (fg)"
    plt.close(fig)
