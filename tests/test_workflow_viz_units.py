"""Units-from-schema wiring + no-double-img-wrap guards for the mass-bearing
visualizations (workflow, multigeneration, v1_v2)."""
import matplotlib
matplotlib.use("Agg")

import v2ecoli.visualizations  # noqa: F401  — import triggers resolver registration
from viva_superpowers.visualization import Visualization


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


# ---------------------------------------------------------------------------
# Task 8 — multigeneration.py + v1_v2.py mass axes
# ---------------------------------------------------------------------------

def test_multigen_and_v1v2_mass_unit_resolves():
    # Both plot dry/cell mass -> fg from the schema.
    assert Visualization.resolve_unit("listeners.mass.dry_mass") == "fg"
    assert Visualization._append_unit("Mass", "fg") == "Mass (fg)"


def test_finalize_drives_dry_mass_label_from_schema():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylabel("Dry Mass")
    Visualization.finalize_figure(fig, [(ax, "y", "listeners.mass.dry_mass")])
    assert ax.get_ylabel() == "Dry Mass (fg)"
    plt.close(fig)


def test_multigeneration_render_no_double_img_wrap():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from v2ecoli.core import build_core
    history = [
        {"generation": 1, "time": 0,   "dry_mass": 100, "protein_mass": 50},
        {"generation": 1, "time": 60,  "dry_mass": 150, "protein_mass": 75},
        {"generation": 2, "time": 60,  "dry_mass": 80,  "protein_mass": 40},
        {"generation": 2, "time": 120, "dry_mass": 130, "protein_mass": 65},
    ]
    viz = MultigenerationVisualization(config={"title": "mg"}, core=build_core())
    html = viz.update({"history": history, "metadata": {}})["html"]
    assert html.count("<img") == html.count("data:image/png")
    assert html.count("<img") >= 1
    assert "base64,<img" not in html


def test_v1v2_render_no_double_img_wrap():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from v2ecoli.core import build_core
    rows = [
        {"time": 0,   "dry_mass": 100, "cell_mass": 300, "protein_mass": 50,
         "volume": 1.0, "instantaneous_growth_rate": 1e-4,
         "n_chromosomes": 1, "n_forks": 0},
        {"time": 120, "dry_mass": 160, "cell_mass": 480, "protein_mass": 80,
         "volume": 1.6, "instantaneous_growth_rate": 1.2e-4,
         "n_chromosomes": 2, "n_forks": 2},
    ]
    viz = V1V2Visualization(config={"title": "cmp"}, core=build_core())
    html = viz.update({"history_v2ecoli": rows, "metadata": {"duration_sec": 120}})["html"]
    assert html.count("<img") == html.count("data:image/png")
    assert html.count("<img") >= 1
    assert "base64,<img" not in html
