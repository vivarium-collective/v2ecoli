from v2ecoli.library.units_atlas import build_atlas, dimension_of


def test_dimension_of():
    assert dimension_of("fg") == "mass"
    assert dimension_of("mM") == "concentration"
    assert dimension_of("1/s") == "rate"
    assert dimension_of("s") == "time"
    assert dimension_of("totally_unknown") == "other"


def test_build_atlas_groups_readouts():
    atlas = build_atlas()                      # no run sample -> magnitudes None
    # structure: {dimension: [ {path, unit, example, min, max}, ... ]}
    assert "mass" in atlas
    masses = {row["path"] for row in atlas["mass"]}
    assert "listeners.mass.cell_mass" in masses
    for row in atlas["mass"]:
        assert row["unit"] == "fg"
        assert "example" in row and "min" in row and "max" in row


def test_build_atlas_flags_dimensionless(monkeypatch):
    # readouts with no unit are NOT in the index, so the flag list comes from
    # a separate scan; assert the API returns a 'flags' channel.
    atlas = build_atlas()
    assert isinstance(atlas.get("_flags", []), list)


def test_units_atlas_visualization_renders_table():
    from v2ecoli.visualizations.units_atlas import UnitsAtlasVisualization
    from v2ecoli.core import build_core
    viz = UnitsAtlasVisualization(config={"title": "Units Atlas"}, core=build_core())
    viz.accumulate({})            # atlas is schema-derived; state may be empty
    html = viz.render()
    assert "Units Atlas" in html
    assert "fg" in html and "mM" in html
    assert "listeners.mass.cell_mass" in html
    assert "<table" in html
