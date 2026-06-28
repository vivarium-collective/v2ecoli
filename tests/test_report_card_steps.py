from process_bigraph.composite import as_step
from v2ecoli.core import build_core
from scripts._compare.report_cards import _sections_to_html, CARD_INPUTS, CARD_OUTPUTS


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
