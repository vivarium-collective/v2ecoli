"""Self-contained inline-SVG plots for report cards (matplotlib, Agg backend).

Each function returns an ``<svg>...</svg>`` string to embed directly in the
report HTML — no external assets, no JS. Two plot kinds:

  - ``violin_strip`` — population stats over the N cells (one point per cell),
    with the pinned reference distribution shown faintly behind.
  - ``loglog_scatter`` — candidate vs reference vector (transcriptome/proteome)
    on log-log axes with the identity line and R^2.
"""
from __future__ import annotations

import io


def _svg(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    # strip the XML/doctype preamble so it embeds inline cleanly
    i = svg.find("<svg")
    return svg[i:] if i >= 0 else svg


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def violin_strip(values, ref_values=None, *, label="", units="", scale=1.0,
                 y_from_zero=False, width=4.2, height=2.6) -> str:
    """Reference vs measured as two **side-by-side** violins (each with a
    jittered per-cell strip). ``scale`` multiplies all values for display
    (e.g. 1/3600 to show seconds as hours); ``y_from_zero`` pins the y-axis
    base at 0."""
    plt = _setup()
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(width, height))

    series = []
    if ref_values and len(ref_values) > 0:
        series.append(("reference", np.asarray(ref_values, float) * scale, "#9aa3af"))
    if values and len(values) > 0:
        series.append(("measured", np.asarray(values, float) * scale, "#1a7f37"))

    positions = list(range(len(series)))
    for pos, (name, data, color) in zip(positions, series):
        if len(data) > 1:
            vp = ax.violinplot([data], positions=[pos], widths=0.7, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color)
                b.set_alpha(0.45 if name == "reference" else 0.22)
                b.set_edgecolor(color)
        x = rng.normal(pos, 0.045, size=len(data))
        ax.scatter(x, data, s=20, color=color, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([s[0] for s in series], fontsize=8)
    ax.set_xlim(-0.6, max(positions, default=0) + 0.6)
    if y_from_zero:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(units or label, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.12)
    return _svg(fig)


def loglog_scatter(cand_vec, ref_vec, *, r2=None, label="",
                   width=3.4, height=3.2) -> str:
    """Candidate vs reference ensemble-mean vector on log-log axes with the
    identity line. Points are per-gene/protein (ensemble means)."""
    plt = _setup()
    import numpy as np
    cand = np.asarray(cand_vec, float)
    ref = np.asarray(ref_vec, float)
    m = (cand > 0) & (ref > 0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(ref[m], cand[m], s=5, alpha=0.25, color="#1f6feb", edgecolor="none")
    if m.any():
        lo = min(ref[m].min(), cand[m].min())
        hi = max(ref[m].max(), cand[m].max())
        ax.plot([lo, hi], [lo, hi], color="#9aa3af", lw=1, ls="--", zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reference", fontsize=9); ax.set_ylabel("candidate", fontsize=9)
    ax.tick_params(labelsize=8)
    if r2 is not None:
        ax.text(0.04, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold", color="#1a1d21")
    ax.set_aspect("equal", adjustable="box")
    return _svg(fig)
