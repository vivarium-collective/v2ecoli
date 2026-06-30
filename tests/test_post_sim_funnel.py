import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_report_cards_funnel_into_post_sim():
    import v2ecoli.workflow.report_cards  # noqa: F401 — triggers card registration
    from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY
    assert POST_SIM_REGISTRY.get("tests") == {
        "cls": __import__("v2ecoli.workflow.report_cards.tests_card",
                          fromlist=["TestsCard"]).TestsCard,
        "kind": "report_card"}
    assert POST_SIM_REGISTRY["vs_vecoli"]["kind"] == "report_card"


def test_analysis_funnels_into_post_sim():
    from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
    from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY

    class _ProbeViz(Analysis):
        name = "probe_viz_demo"
        scale = "single"
        def analyze(self, **kw):
            return {"view": "<i></i>", "data": {}}

    assert POST_SIM_REGISTRY["probe_viz_demo"]["kind"] == "analysis"
    # back-compat: the legacy registry still has it too
    assert ANALYSIS_REGISTRY["probe_viz_demo"] is _ProbeViz
