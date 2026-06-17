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
                 ref_label="reference", meas_label="measured",
                 y_from_zero=False, width=4.2, height=2.6) -> str:
    """Reference vs measured as two **side-by-side** violins (each with a
    jittered per-cell strip). ``scale`` multiplies all values for display
    (e.g. 1/3600 to show seconds as hours); ``y_from_zero`` pins the y-axis
    base at 0. ``ref_label``/``meas_label`` name the two series (e.g. v1/v2)."""
    plt = _setup()
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(width, height))

    series = []
    if ref_values and len(ref_values) > 0:
        series.append((ref_label, np.asarray(ref_values, float) * scale, "#9aa3af", True))
    if values and len(values) > 0:
        series.append((meas_label, np.asarray(values, float) * scale, "#1a7f37", False))

    positions = list(range(len(series)))
    for pos, (name, data, color, is_ref) in zip(positions, series):
        if len(data) > 1:
            vp = ax.violinplot([data], positions=[pos], widths=0.7, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color)
                b.set_alpha(0.45 if is_ref else 0.22)
                b.set_edgecolor(color)
        x = rng.normal(pos, 0.045, size=len(data))
        ax.scatter(x, data, s=20, color=color, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([s[0] for s in series], fontsize=8)
    ax.set_xlim(-0.6, max(positions, default=0) + 0.6)
    ax.set_ylabel(units or label, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.12)
    if y_from_zero:
        # include zero in range, whichever side the (one-sided) data is on:
        # positive data -> floor at 0; negative (uptake) -> ceil at 0.
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0.0, lo), max(0.0, hi))
        ax.axhline(0, color="#d0d4d9", lw=0.6, zorder=0)
    return _svg(fig)


def _cite(sid: str) -> str:
    """``firstauthor_year`` source_id -> 'Firstauthor Year' for display."""
    parts = str(sid).split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{parts[0].capitalize()} {parts[-1]}"
    return str(sid)


def literature_strip(sim_values, sim_mean, measured, *, measured_unc=None,
                     labels=None, theoretical=None, theoretical_label=None,
                     units="", label="", scale=1.0, width=4.8, height=2.9) -> str:
    """vs_literature axis: the simulated population vs curated experiment.

    Colour scheme: **green = sim** (a violin + jittered strip of the per-cell
    population when given, plus a diamond at the graded mean), **black =
    experimental measurements** (one marker per source, with ± error bars where
    an uncertainty is given), **red dashed line = theoretical limit** (a ceiling
    the model should not cross). Each experimental source sits at its own x
    position so the spread across studies is visible."""
    plt = _setup()
    import numpy as np
    GREEN, BLACK, RED = "#1a7f37", "#1a1d21", "#c62828"
    fig, ax = plt.subplots(figsize=(width, height))
    measured = list(measured or [])
    unc = list(measured_unc) if measured_unc else [None] * len(measured)
    labels = list(labels) if labels else [f"src{i+1}" for i in range(len(measured))]

    # theoretical limit — a red dashed horizontal ceiling spanning the axis,
    # labelled with its source citation
    if theoretical is not None:
        cite = f"{_cite(theoretical_label)} · " if theoretical_label else ""
        ax.axhline(theoretical * scale, color=RED, lw=1.5, ls="--", zorder=1,
                   label=f"theoretical limit — {cite}{theoretical * scale:.3g}")

    # sim (green): violin + strip of the per-cell population, then a mean diamond
    if sim_values and len(sim_values) > 1:
        data = np.asarray(sim_values, float) * scale
        vp = ax.violinplot([data], positions=[0], widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(GREEN); b.set_alpha(0.28); b.set_edgecolor(GREEN)
        jit = np.random.default_rng(0).normal(0, 0.05, size=len(data))
        ax.scatter(jit, data, s=14, color=GREEN, alpha=0.65, edgecolor="white",
                   linewidth=0.4, zorder=3)
    if sim_mean is not None:
        ax.scatter([0], [sim_mean * scale], s=95, marker="D", color=GREEN,
                   edgecolor="white", linewidth=0.8, zorder=5, label="sim (v2ecoli)")

    # experimental (black): one marker per source, ± uncertainty where present
    xs = list(range(1, len(measured) + 1))
    for xi, m, u in zip(xs, measured, unc):
        if u:
            ax.errorbar([xi], [m * scale], yerr=[u * scale], fmt="o", color=BLACK,
                        ms=5, capsize=3, elinewidth=1, zorder=4)
        else:
            ax.scatter([xi], [m * scale], s=34, color=BLACK, zorder=4)
    if measured:
        ax.scatter([], [], s=34, color=BLACK, label="experimental")  # legend proxy

    ax.set_xticks([0] + xs)
    ax.set_xticklabels(["sim"] + [_cite(l) for l in labels], fontsize=7,
                       rotation=30, ha="right")
    ax.set_xlim(-0.7, len(measured) + 0.7)
    ax.set_ylabel(units or label, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.15)
    ax.legend(fontsize=6.5, loc="best", frameon=False)
    return _svg(fig)


def flux_scatter(cand_vec, ref_vec, *, ids=None, r2=None, active_eps=1e-6,
                 qual_eps=1e-3, ref_std=None, cand_std=None, label="", width=3.8,
                 height=3.4, ref_label="reference", meas_label="candidate") -> str:
    """Exchange-flux candidate vs reference on **symlog** signed axes (fluxes
    span orders of magnitude and are signed: negative=uptake, positive=
    secretion). Pairs that are inactive in both (|both|<eps) are excluded.
    Matched pairs are blue (with x/y error bars = cell-to-cell std if given);
    an exchange that *appeared* (ref~0, cand active) is red and one that
    *disappeared* (active, cand~0) is orange — both labelled, since a
    qualitative change is the big regression flag."""
    plt = _setup()
    import numpy as np
    cand = np.asarray(cand_vec, float)
    ref = np.asarray(ref_vec, float)
    rstd = np.asarray(ref_std, float) if ref_std is not None else None
    cstd = np.asarray(cand_std, float) if cand_std is not None else None
    ids = list(ids) if ids is not None else [str(i) for i in range(len(cand))]
    ca, ra = np.abs(cand) > active_eps, np.abs(ref) > active_eps
    matched = ca & ra
    appeared = ca & ~ra
    disappeared = ~ca & ra
    # A flip whose active-side magnitude is below qual_eps is near-floor jitter
    # (shown, not graded as a qualitative change) — draw it muted, not red/orange.
    sub = ((appeared & (np.abs(cand) < qual_eps)) |
           (disappeared & (np.abs(ref) < qual_eps)))
    appeared = appeared & ~sub
    disappeared = disappeared & ~sub
    fig, ax = plt.subplots(figsize=(width, height))
    if rstd is not None and cstd is not None:
        ax.errorbar(ref[matched], cand[matched], xerr=rstd[matched],
                    yerr=cstd[matched], fmt="none", ecolor="#1f6feb",
                    elinewidth=0.6, alpha=0.35, zorder=2)
    ax.scatter(ref[matched], cand[matched], s=14, alpha=0.6, color="#1f6feb",
               edgecolor="none", zorder=3)
    keep = matched | appeared | disappeared | sub
    if keep.any():
        vals = np.concatenate([ref[keep], cand[keep]])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.15 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#9aa3af", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="#d0d4d9", lw=0.6, zorder=0)
        ax.axvline(0, color="#d0d4d9", lw=0.6, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for flag, color, size, ann, name in (
            (appeared, "#c62828", 26, True, "appeared"),
            (disappeared, "#ef6c00", 26, True, "lost"),
            (sub, "#9aa3af", 14, False, "below floor")):
        idxs = np.where(flag)[0]
        if not len(idxs):
            continue
        ax.scatter(ref[idxs], cand[idxs], s=size, color=color, edgecolor="white",
                   linewidth=0.5, zorder=4, label=f"{name} ({len(idxs)})")
        if ann:
            for i in idxs:
                ax.annotate(ids[i][:-3] if ids[i].endswith("]") else ids[i],
                            (ref[i], cand[i]), fontsize=6.5, color=color,
                            xytext=(3, 3), textcoords="offset points")
    lin = max(abs(lo), abs(hi)) * 1e-3 if keep.any() else 1e-3
    ax.set_xscale("symlog", linthresh=lin); ax.set_yscale("symlog", linthresh=lin)
    ax.set_xlabel(f"{ref_label} flux", fontsize=9)
    ax.set_ylabel(f"{meas_label} flux", fontsize=9)
    ax.tick_params(labelsize=7)
    if r2 is not None:
        ax.text(0.04, 0.96, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold", color="#1a1d21")
    if appeared.any() or disappeared.any():
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    return _svg(fig)


def generation_trend(by_cell, ref_by_cell=None, *, scale=1.0, units="", label="",
                     rho=None, ref_rho=None, y_from_zero=False,
                     ref_label="reference", meas_label="measured",
                     width=4.2, height=2.8) -> str:
    """Per-lineage metric vs generation — the companion plot for a flagged
    generation-drift. Mirrors the violin's reference‖measured convention:
    **measured lineages in green, reference lineages in grey**, one connected
    line per lineage (seed; lineages aren't individually labelled). ``by_cell``
    and ``ref_by_cell`` are ``[[seed, gen, value], ...]``. The Spearman ρ
    annotation names the dataset it summarizes (the flag is driven by the
    measured series)."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(width, height))

    def _draw(points, color, zorder, lbl):
        if not points:
            return
        bylin: dict = {}
        for sd, gg, v in points:
            bylin.setdefault(sd, []).append((gg, v * scale))
        first = True
        for sd in sorted(bylin):
            pts = sorted(bylin[sd])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                    lw=1.3, marker="o", ms=3, alpha=0.85, zorder=zorder,
                    label=(lbl if first else None))
            first = False

    _draw(ref_by_cell, "#9aa3af", 2, ref_label)     # grey, matches the violin
    _draw(by_cell, "#1a7f37", 3, meas_label)        # green, matches the violin
    ax.set_xlabel("generation", fontsize=9)
    ax.set_ylabel(units or label, fontsize=9)
    allg = sorted({c[1] for c in (by_cell or [])} | {c[1] for c in (ref_by_cell or [])})
    if allg:
        ax.set_xticks(allg)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lines = []
    if rho is not None:
        lines.append(f"{meas_label} ρ(gen) = {rho:+.2f}")
    if ref_rho is not None:
        lines.append(f"{ref_label} ρ(gen) = {ref_rho:+.2f}")
    if lines:
        ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                va="top", fontweight="bold", color="#ef6c00")
    if by_cell or ref_by_cell:
        ax.legend(fontsize=7, loc="best", frameon=False, ncol=2)
    ax.margins(y=0.12)
    if y_from_zero:
        ax.set_ylim(bottom=0)        # match the violin's y-axis (from 0)
    return _svg(fig)


def loglog_scatter(cand_vec, ref_vec, *, r2=None, label="",
                   width=3.4, height=3.2,
                   ref_label="reference", meas_label="candidate") -> str:
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
    ax.set_xlabel(ref_label, fontsize=9); ax.set_ylabel(meas_label, fontsize=9)
    ax.tick_params(labelsize=8)
    if r2 is not None:
        ax.text(0.04, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold", color="#1a1d21")
    ax.set_aspect("equal", adjustable="box")
    return _svg(fig)
