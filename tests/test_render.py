import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.workflow.render import fig_to_html


def test_fig_to_html_embeds_svg():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    html = fig_to_html(fig, title="Demo")
    assert "<svg" in html
    assert "Demo" in html
    assert html.strip().startswith("<")
