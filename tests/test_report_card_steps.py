from process_bigraph.composite import as_step
from v2ecoli.core import build_core
from scripts._compare.report_cards import _sections_to_html, CARD_INPUTS, CARD_OUTPUTS
from _card_helpers import _state, _run_card


def test_sections_to_html_renders_html_and_rows():
    html = _sections_to_html([
        {"title": "A", "html": "<b>hi</b>"},
        {"title": "B", "rows": [{"label": "x", "left": "1", "right": "2",
                                 "verdict": "within_tol", "reason": "ok"}]},
    ])
    assert html.lstrip().startswith("<")
    assert "<b>hi</b>" in html and "A" in html
    assert "x" in html and "within_tol" in html        # rows rendered as a table


def test_as_step_card_round_trips_through_core():
    @as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="probe_report_card",
             aliases=["probe"])
    def update_probe_report_card(state):
        return {"card_html": f"<p>{state['name']}</p>", "verdict": "within_tol", "axes": []}
    core = build_core()
    core.register_link("probe_report_card", update_probe_report_card)
    StepCls = core.link_registry["probe_report_card"]
    step = StepCls(config={}, core=core)
    out = step.update({"name": "basal", "condition": "basal", "seeds": 1,
                       "generations": 4, "variant": 0, "observables": {},
                       "plot_trajs": {}, "v2_bounds": [], "config": {},
                       "v2_dir": "", "ve_dir": ""})
    assert out["verdict"] == "within_tol" and "basal" in out["card_html"]


# 5 seeds so the t-test / median have data; one within-tol observable.
_PO = {"rna_mass": [{"median_rel": 0.02, "max_rel": 0.05, "init_ve": 100.0,
                     "init_v2": 101.0, "init_t": 60.0, "ve_mean": 100.0, "v2_mean": 101.0}
                    for _ in range(5)]}


def test_standard_step_grades_and_renders():
    out = _run_card("standard", _state(_PO))
    assert out["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    assert any(a["id"].startswith("standard.") for a in out["axes"])
    assert "<" in out["card_html"]


def test_parca_step_grades_initial_state():
    out = _run_card("parca", _state(_PO))
    assert out["verdict"] == "within_tol"
    assert any(a["id"].startswith("parca.") for a in out["axes"])


def test_statistical_step_grades():
    out = _run_card("statistical", _state(_PO, name="statistical", seeds=4))
    assert out["verdict"] == "within_tol"
    assert out["axes"]


def test_config_step_is_ungraded_and_renders_config():
    out = _run_card("config", _state({}, name="basal", config={"condition": "basal"}))
    assert out["verdict"] == "ungraded" and out["axes"] == []
    assert "basal" in out["card_html"]
