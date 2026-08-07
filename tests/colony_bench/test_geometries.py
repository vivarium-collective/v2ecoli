import pytest


@pytest.mark.parametrize("builder,kwargs", [
    ("free_colony", dict(n_cells=2)),
    ("mother_machine", dict(n_channels=3)),
    ("daughter_machine", dict()),
])
def test_geometry_builds_simple_document(builder, kwargs):
    from v2ecoli.colony_bench import geometries
    doc = getattr(geometries, builder)("simple", seed=0, **kwargs)
    assert "cells" in doc and len(doc["cells"]) >= 1
    assert doc["multibody"]["address"] == "local:PymunkProcess"
    # every cell carries the simple-tier division process
    for cell in doc["cells"].values():
        assert "grow_divide" in cell


def test_mother_machine_has_barriers_and_removal():
    from v2ecoli.colony_bench import geometries
    doc = geometries.mother_machine("simple", n_channels=4, seed=0)
    assert doc["multibody"]["config"]["barriers"]
    assert "remove_crossing" in doc
    assert len(doc["cells"]) == 4  # one cell per channel
