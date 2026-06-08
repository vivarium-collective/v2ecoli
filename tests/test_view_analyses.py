"""Tests for Analysis subclasses that return a rendered HTML view."""

from bigraph_schema import allocate_core  # noqa: F401


def test_mass_fraction_voronoi_registered_single_view():
    from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["mass_fraction_voronoi"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ccm_scatter_registered_multiseed():
    from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["central_carbon_metabolism_scatter"]
    assert issubclass(cls, Analysis) and cls.scale == "multiseed"
