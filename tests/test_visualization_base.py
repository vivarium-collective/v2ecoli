import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.steps.base import V2Step
from v2ecoli.workflow.post_sim import POST_SIM_REGISTRY, Visualization


class _DemoViz(Visualization):
    name = "demo_viz"
    def render(self, study):
        return "<div>viz</div>", {"k": 1}


def test_visualization_ports_and_registration(core):
    v = _DemoViz({}, core=core)
    assert isinstance(v, V2Step)
    assert v.outputs() == {"view": "string", "data": "map"}
    assert POST_SIM_REGISTRY["demo_viz"]["kind"] == "visualization"


def test_visualization_update_returns_view_and_data(core):
    out = _DemoViz({}, core=core).update({"study": object()})
    assert out["view"] == "<div>viz</div>"
    assert out["data"] == {"k": 1}


def test_visualization_render_none_yields_empty(core):
    class _Empty(Visualization):
        name = "empty_viz"
        def render(self, study):
            return None
    out = _Empty({}, core=core).update({"study": None})
    assert out == {"view": "", "data": {}}
