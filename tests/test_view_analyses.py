"""Tests for Analysis subclasses that return a rendered HTML view."""

from bigraph_schema import allocate_core  # noqa: F401


def test_mass_fraction_voronoi_registered_single_view():
    from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["mass_fraction_voronoi"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
