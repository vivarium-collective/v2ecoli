"""House plotting style for the v2ecoli pbg investigation.

This module encodes the chart conventions that the expert reviewer (Rashmi)
re-requested ~7 times across the dnaa investigation, so future figures stop
re-incurring that friction. The conventions, verbatim from the feedback:

  1. Time axis in MINUTES (not seconds/ticks).
  2. Generation / lineage boundaries marked with vertical demarcation lines.
  3. Axis labels ALWAYS carry units.
  4. No stray gridlines — only intentional lineage/threshold lines.
  5. Avoid overlapping labels.
  6. Occupancy / band plots focus on STEADY-STATE generations (skip early transient).
  7. Long figure subtitles must WRAP (not clip the iframe edge).

It centralizes the minutes-stitching + subtitle-wrapping + house-layout logic that
was previously duplicated inline in scripts/render_dnaa4_autoreg._frame,
scripts/render_dnaa5_validate, and scripts/make_dnaa5_figures (_wrap/_layout).

This SHOULD graduate to pbg-superpowers as the framework-wide default plotting
style once it has settled, so every pbg investigation inherits the same
expert-approved chart conventions instead of re-deriving them per-workspace.

Import-light by design: only numpy is required at import time; matplotlib and
plotly are imported lazily inside the helpers that need them, so importing this
module does not force either backend.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Shared palette — neutral grey + a small categorical set, matching the dnaa figures.
PALETTE = {
    "grey": "#94a3b8",
    "green": "#16a34a",
    "red": "#dc2626",
    "blue": "#2563eb",
    "amber": "#d97706",
    "purple": "#a855f7",
    "orange": "#ea580c",
    "sky": "#0ea5e9",
    "band_green": "rgba(34,197,94,0.10)",
    "band_blue": "rgba(37,99,235,0.08)",
    "axis": "#475569",
    "muted": "#64748b",
    "spine": "#cbd5e1",
}


# --------------------------------------------------------------------------- #
# Lineage time helpers (convention 1 + 2: minutes + generation demarcation)
# --------------------------------------------------------------------------- #
def stitch_minutes(generation: Sequence, global_time: Sequence) -> np.ndarray:
    """Stitch per-generation global_time (which RESETS to 0 each generation) into
    cumulative lineage time in MINUTES.

    A multi-generation lineage run emits ``global_time`` that resets to 0 at the
    start of each generation; plotting it directly overlays every cell cycle on
    one ~60-min axis (the "two green trajectories" artifact). This lays the
    generations out SEQUENTIALLY and converts seconds -> minutes.

    Args:
        generation: per-sample generation index (int-like).
        global_time: per-sample time in seconds, resetting each generation.

    Returns:
        numpy array of cumulative lineage time in minutes, aligned to the inputs.
    """
    gen = np.asarray(generation, dtype=int)
    gt = np.asarray(global_time, dtype=float)
    t_min = np.zeros_like(gt)
    offset = 0.0
    for g in sorted(set(gen.tolist())):
        m = gen == g
        t_min[m] = (offset + gt[m]) / 60.0
        offset += gt[m].max()
    return t_min


def gen_boundaries(generation: Sequence, t_min: Sequence) -> List[float]:
    """Return the t_min values at which the generation index increments — the
    locations for vertical lineage demarcation lines (convention 2).

    Args:
        generation: per-sample generation index (int-like), same length as t_min.
        t_min: per-sample cumulative lineage time in minutes (see stitch_minutes).

    Returns:
        list of t_min values where a new generation begins (excluding the first).
    """
    gen = np.asarray(generation, dtype=int)
    tm = np.asarray(t_min, dtype=float)
    boundaries: List[float] = []
    prev = None
    for g, t in zip(gen.tolist(), tm.tolist()):
        if prev is not None and g != prev:
            boundaries.append(t)
        prev = g
    return boundaries


# --------------------------------------------------------------------------- #
# Subtitle wrapping (convention 7: long subtitles must wrap, not clip)
# --------------------------------------------------------------------------- #
def wrap(text: str, width: int = 78) -> str:
    """Wrap a subtitle into <br>-separated lines so it doesn't overflow/clip the
    iframe edge (convention 7). Breaks only at word boundaries."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


# --------------------------------------------------------------------------- #
# Matplotlib helpers (conventions 3 + 4: unit-bearing labels, no stray gridlines)
# --------------------------------------------------------------------------- #
def style_axes(ax, xlabel: str, ylabel: str):
    """Apply the house style to a matplotlib Axes: no gridlines, clean spines,
    and unit-bearing axis labels (the caller is responsible for including units
    in xlabel/ylabel — convention 3). Returns the axes for chaining."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)  # convention 4: no stray gridlines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["spine"])
    ax.tick_params(colors=PALETTE["axis"])
    return ax


def mark_lineages(ax, boundaries: Sequence[float]):
    """Draw faint vertical demarcation lines at generation boundaries
    (convention 2) on a matplotlib Axes. Returns the axes for chaining."""
    for b in boundaries:
        ax.axvline(b, color=PALETTE["muted"], lw=0.8, ls=":", alpha=0.5, zorder=0)
    return ax


# --------------------------------------------------------------------------- #
# Plotly helper (conventions 3, 4, 7 bundled into one house layout)
# --------------------------------------------------------------------------- #
def house_layout(fig, title: str, subtitle: str, height: int = 480):
    """Apply the house Plotly layout: wrapped subtitle (convention 7), plotly_white
    template (no stray gridding beyond the clean default), consistent fonts /
    margins / legend, and a height bump proportional to the wrapped subtitle's
    line count so multi-line subtitles don't crowd the plot. Behavior is identical
    to make_dnaa5_figures._layout. Returns the figure for chaining."""
    sub = wrap(subtitle)
    nlines = sub.count("<br>") + 1
    # Count the legend entries so we can reserve the right amount of vertical
    # space below the plot for a VERTICAL legend (one entry per row).
    #   - horizontal legends crammed long labels on top of each other at narrow
    #     embed widths (Rashmi 2026-06-24);
    #   - outside-top-right legends got clipped when labels exceeded the right
    #     margin (earlier report).
    # A vertical legend BELOW the plot, left-aligned, can't overlap (one per row)
    # and can't be right-clipped (it lives within the plot's own width). We size
    # the bottom margin + height to the actual entry count so nothing is cut off
    # and simple 2-3 entry plots don't get excess whitespace.
    n_leg = 0
    for _t in fig.data:
        if getattr(_t, "name", None) and getattr(_t, "showlegend", None) is not False:
            n_leg += 1
    n_leg = max(1, n_leg)
    row_px = 22
    legend_block = 46 + row_px * n_leg          # title-axis gap + one row per entry
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#475569'>{sub}</span>",
            x=0.5, xanchor="center", font=dict(size=17),
            # pin to the container top so the subtitle grows downward into the
            # reserved top margin (not into the plot / subplot titles below it)
            yref="container", y=0.965, yanchor="top"),
        template="plotly_white",
        height=height + 18 * max(0, nlines - 1) + legend_block,
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(t=92 + 22 * nlines, b=64 + row_px * n_leg, l=70, r=50),
        hovermode="x unified",
        legend=dict(yanchor="top", y=-0.16, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="#e2e8f0", borderwidth=1),
    )
    return fig
