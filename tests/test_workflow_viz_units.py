"""Units-from-schema wiring + no-double-img-wrap guards for the mass-bearing
visualizations (workflow, multigeneration, v1_v2)."""
import matplotlib
matplotlib.use("Agg")

import v2ecoli.visualizations  # noqa: F401  — import triggers resolver registration
from pbg_superpowers.visualization import Visualization


# ---------------------------------------------------------------------------
# Task 7 — workflow.py mass axis
# ---------------------------------------------------------------------------

def test_mass_axis_gets_fg_from_schema():
    # The resolver maps listeners.mass.cell_mass -> fg.
    assert Visualization.resolve_unit("listeners.mass.cell_mass") == "fg"
    # _append_unit drives the label; integration is exercised by the viz call.
    assert Visualization._append_unit("Mass", "fg") == "Mass (fg)"


def test_finalize_figure_drives_mass_label_from_schema():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    Visualization.finalize_figure(fig, [(ax, "y", "listeners.mass.cell_mass")])
    assert ax.get_ylabel() == "Mass (fg)"
    plt.close(fig)


def test_plot_mass_returns_single_img_no_double_wrap():
    from v2ecoli.visualizations.workflow import _plot_mass
    history = [
        {"time": 0,  "global_time": 0,  "dry_mass": 100, "protein_mass": 50},
        {"time": 60, "global_time": 60, "dry_mass": 150, "protein_mass": 75},
    ]
    out = _plot_mass(history, title="t")
    assert out.startswith('<img src="data:image/png;base64,')
    assert out.count("<img") == 1
    assert out.count("data:image/png") == 1


def test_workflow_render_has_no_double_img_wrap():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from v2ecoli.core import build_core
    history = [
        {"time": 0,  "global_time": 0,  "dry_mass": 100, "protein_mass": 50},
        {"time": 60, "global_time": 60, "dry_mass": 150, "protein_mass": 75},
    ]
    viz = WorkflowVisualization(config={"title": "wf"}, core=build_core())
    html = viz.update({"history": history, "metadata": {}})["html"]
    # Exactly one <img> per data URI everywhere — no nested <img src="...<img...">.
    assert html.count("<img") == html.count("data:image/png")
    assert html.count("<img") >= 1
    assert "<img src=\"data:image/png;base64,<img" not in html
