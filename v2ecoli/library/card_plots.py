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
                     units="", label="", scale=1.0, width=4.8, height=3.2) -> str:
    """vs_literature axis: the simulated population vs curated experiment.

    Two kinds of spread, side by side. **Green = sim** (a violin + jittered strip
    of the per-cell population + a diamond at the graded mean) is cell-to-cell
    biological heterogeneity. **Black = measured** is a single strip of one dot per
    literature source (± its reported uncertainty), read as the spread of point
    estimates ACROSS studies — collapsed to one x position (not a tick per source)
    so the axis stays readable as more datapoints accrue; a faint vertical bar marks
    the measured min–max band the criterion grades against, and the per-source
    values are listed in the caption for provenance. **Red dashed line = theoretical
    limit** (a ceiling the model should not cross)."""
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
                   edgecolor="white", linewidth=0.8, zorder=5,
                   label="sim (v2ecoli) — cell-to-cell")

    # measured (black): ONE strip at x=1, a dot per study (jittered so equal values
    # separate), ± uncertainty; a faint bar marks the graded min–max band. NOT a
    # KDE violin — a handful of study point estimates is a dotplot, not a density.
    if measured:
        ms = np.asarray(measured, float) * scale
        lo, hi = float(ms.min()), float(ms.max())
        if hi > lo:
            ax.plot([1, 1], [lo, hi], color=BLACK, lw=4, alpha=0.12, zorder=2,
                    solid_capstyle="round")
        jit = (np.random.default_rng(1).normal(0, 0.045, size=len(ms))
               if len(ms) > 1 else np.zeros(len(ms)))
        for xj, m, u in zip(1 + jit, ms, unc):
            if u:
                ax.errorbar([xj], [m], yerr=[u * scale], fmt="o", color=BLACK,
                            ms=5, capsize=2.5, elinewidth=1, zorder=4)
            else:
                ax.scatter([xj], [m], s=34, color=BLACK, zorder=4)
        n = len(measured)
        ax.scatter([], [], s=34, color=BLACK,
                   label=f"measured — {n} stud{'y' if n == 1 else 'ies'}")

    ax.set_xticks([0, 1] if measured else [0])
    ax.set_xticklabels(["sim", "measured"] if measured else ["sim"], fontsize=8)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel(units or label, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.15)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(0.0, lo), max(0.0, hi))   # pin the y-axis floor to 0 (match v1↔v2 violins)
    ax.legend(fontsize=6.5, loc="best", frameon=False)

    # provenance caption — keep which study reported what, since the dots no longer
    # carry per-source x-ticks.
    if measured:
        prov = "measured: " + " · ".join(
            f"{_cite(l)} {m * scale:.3g}" for l, m in zip(labels, measured))
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.5, 0.015, prov, ha="center", va="bottom", fontsize=6.2,
                 color="#555")
    return _svg(fig)


def flux_scatter(cand_vec, ref_vec, *, ids=None, r2=None, active_eps=1e-6,
                 qual_eps=1e-3, qualitative=True, ref_std=None, cand_std=None,
                 label="", width=3.8, height=3.4, ref_label="reference",
                 meas_label="candidate") -> str:
    """Flux candidate vs reference on **symlog** signed axes (fluxes span orders
    of magnitude and are signed). Pairs inactive in both (|both|<eps) are excluded.
    Matched pairs are blue (with x/y error bars = cell-to-cell std if given). With
    ``qualitative=True`` (exchange fingerprints) an exchange that *appeared*
    (ref~0, cand active) is red and one that *disappeared* (active, cand~0) is
    orange — a metabolite switching on/off is the regression flag. With
    ``qualitative=False`` (internal fluxes, where a 0 is just a low value, not a
    categorical on/off) every point is drawn uniformly: only position vs the
    identity line and sign carry meaning."""
    plt = _setup()
    import numpy as np
    cand = np.asarray(cand_vec, float)
    ref = np.asarray(ref_vec, float)
    rstd = np.asarray(ref_std, float) if ref_std is not None else None
    cstd = np.asarray(cand_std, float) if cand_std is not None else None
    ids = list(ids) if ids is not None else [str(i) for i in range(len(cand))]
    ca, ra = np.abs(cand) > active_eps, np.abs(ref) > active_eps
    if qualitative:
        matched = ca & ra
        appeared = ca & ~ra
        disappeared = ~ca & ra
        # A flip whose active-side magnitude is below qual_eps is near-floor jitter
        # (shown, not graded as a qualitative change) — draw it muted, not red/orange.
        sub = ((appeared & (np.abs(cand) < qual_eps)) |
               (disappeared & (np.abs(ref) < qual_eps)))
        appeared = appeared & ~sub
        disappeared = disappeared & ~sub
    else:
        # No on/off semantics: keep every pair active on either side, all blue.
        matched = ca | ra
        appeared = disappeared = sub = np.zeros(len(cand), dtype=bool)
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


def loglog_scatter(cand_vec, ref_vec, *, r2=None, stat_label="R²", label="",
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
        ax.text(0.04, 0.95, f"{stat_label} = {r2:.4f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold", color="#1a1d21")
    ax.set_aspect("equal", adjustable="box")
    return _svg(fig)


def ternary_plot(branches: dict, influx=None, *, ref_fractions=None,
                 ref_label="reference", extra_refs=None, residual_max=0.05,
                 label="", meas_label="measured", width=4.0, height=3.6) -> str:
    """A 2-simplex (ternary) plot of a 3-branch flux composition.

    ``branches`` is the measured ``{name: flux}`` (3 keys); the model point is
    its renormalized composition. ``ref_fractions`` is the reference composition
    (same keys); ``extra_refs`` an optional list of ``(label, {name: frac})`` for
    additional reference points. A companion bar shows the closure residual
    (= 1 − Σbranch/influx, the node influx unaccounted by the three branches —
    expected small, a biomass drain) against ``residual_max``."""
    plt = _setup()
    import numpy as np
    names = list(branches.keys())[:3]
    if len(names) != 3:
        return ""
    # 2-simplex corners: top = names[0], bottom-left = names[1], bottom-right = names[2]
    corners = {0: np.array([0.5, 1.0]), 1: np.array([0.0, 0.0]), 2: np.array([1.0, 0.0])}

    def _xy(frac):  # barycentric (3 fractions summing to 1) -> xy
        f = [frac.get(n, 0.0) for n in names]
        s = sum(f) or 1.0
        f = [x / s for x in f]
        return f[0] * corners[0] + f[1] * corners[1] + f[2] * corners[2]

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [3.2, 1]})
    # gridlines: for each component, lines of constant fraction (10% spacing)
    # run parallel to the opposite edge — the standard ternary grid.
    for f in [i / 10 for i in range(1, 10)]:
        for k in range(3):
            a = f * corners[k] + (1 - f) * corners[(k + 1) % 3]
            b = f * corners[k] + (1 - f) * corners[(k + 2) % 3]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#e9ecef", lw=0.5, zorder=0)
    # triangle edges
    tri = np.array([corners[0], corners[1], corners[2], corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#9aa3af", lw=1.2, zorder=1)
    for k, n in enumerate(names):
        c = corners[k]
        ax.annotate(n, c, fontsize=9, fontweight="bold", ha="center",
                    va="bottom" if k == 0 else "top",
                    xytext=(0, 7 if k == 0 else -7), textcoords="offset points")
    # scale ticks for the top component (EMP) along the two upper edges
    for f in [i / 10 for i in range(2, 9, 2)]:
        pL = f * corners[0] + (1 - f) * corners[1]   # EMP–oxPPP edge
        ax.annotate(f"{f:.0%}", pL, fontsize=5.5, color="#9aa3af", ha="right",
                    va="center", xytext=(-2, 0), textcoords="offset points", zorder=1)
    # reference point(s)
    refs = []
    if ref_fractions:
        refs.append((ref_label, ref_fractions, "#111827", "o"))
    for lbl, fr in (extra_refs or []):
        refs.append((lbl, fr, "#9aa3af", "s"))
    for lbl, fr, col, mk in refs:
        p = _xy(fr)
        ax.scatter(*p, s=55, color=col, marker=mk, zorder=4, edgecolor="white",
                   linewidth=0.6, label=lbl)
    # model point, annotated with its split
    mp = _xy(branches)
    ax.scatter(*mp, s=90, color="#1a7f37", marker="*", zorder=5,
               edgecolor="white", linewidth=0.6, label=meas_label)
    tot = sum(branches.get(n, 0.0) for n in names) or 1.0
    split = " / ".join(f"{100 * branches.get(n, 0.0) / tot:.0f}" for n in names)
    ax.annotate(f"{split}%", mp, fontsize=7, color="#1a7f37", ha="left", va="center",
                xytext=(8, 0), textcoords="offset points", zorder=6)
    ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0)
    ax.text(0.5, -0.13, "gridlines 10% · toward a corner = more of that branch",
            transform=ax.transAxes, ha="center", fontsize=6, color="#9aa3af")

    # closure-residual companion bar
    resid = None
    if influx:
        resid = 1.0 - sum(branches.get(n, 0.0) for n in names) / influx
    axr.axhspan(0, 100 * residual_max, color="#1a7f37", alpha=0.12)
    if resid is not None:
        col = "#1a7f37" if abs(resid) <= residual_max else (
            "#ef6c00" if abs(resid) <= 2 * residual_max else "#c62828")
        axr.bar([0], [100 * resid], width=0.6, color=col)
        axr.annotate(f"{100 * resid:+.1f}%", (0, 100 * resid), ha="center",
                     va="bottom" if resid >= 0 else "top", fontsize=8,
                     xytext=(0, 3 if resid >= 0 else -3), textcoords="offset points")
    axr.axhline(0, color="#6b7280", lw=0.8)
    axr.set_xlim(-0.6, 0.6); axr.set_xticks([])
    axr.set_title("closure\nresidual", fontsize=8)
    axr.set_ylabel("% of influx", fontsize=8)
    axr.tick_params(labelsize=7)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    return _svg(fig)


def composition_bars(branches: dict, ref_fractions=None, *, influx=None, label="",
                     meas_label="model", ref_label="reference", xlabel="fraction of node flux",
                     width=4.4, height=2.2) -> str:
    """Two stacked horizontal bars — model over reference — of a branch-point
    flux composition, each renormalized to 1. For 2- to N-way fate splits where a
    ternary doesn't apply (e.g. isocitrate ICDH/ICL, or AcCoA TCA/acetate/
    biosynthesis). ``branches`` is the measured ``{name: flux}``; ``ref_fractions``
    the reference composition (same keys). When ``influx`` is given, the model's
    branches are taken as fractions of influx and any shortfall shows as a hatched
    residual tail. Segments ≥ 6% are labelled with their %."""
    plt = _setup()
    names = list(ref_fractions or branches)
    bsum = sum(branches.get(n, 0.0) for n in names)
    if influx:
        mfr = [branches.get(n, 0.0) / influx for n in names]
        resid = max(0.0, 1.0 - sum(mfr))
    else:
        mfr = [branches.get(n, 0.0) / (bsum or 1.0) for n in names]
        resid = 0.0
    rsum = sum((ref_fractions or {}).values()) or 1.0
    rfr = [ref_fractions[n] / rsum for n in names] if ref_fractions else None
    COLORS = ["#1a7f37", "#1f6feb", "#e8893b", "#9b59b6", "#16a2a2", "#b0b6bd"]
    cmap = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}
    fig, ax = plt.subplots(figsize=(width, height))
    rows = [(meas_label, mfr, resid)]
    if rfr is not None:
        rows.append((ref_label, rfr, 0.0))
    for yi, (lab, fr, rs) in enumerate(rows):
        left = 0.0
        for n, f in zip(names, fr):
            if f <= 0:
                continue
            ax.barh(yi, f, left=left, color=cmap[n], edgecolor="white",
                    height=0.62, zorder=2)
            if f >= 0.06:
                ax.text(left + f / 2, yi, f"{100 * f:.0f}%", ha="center",
                        va="center", fontsize=7, color="white", fontweight="bold")
            left += f
        if rs > 0.01:
            ax.barh(yi, rs, left=left, color="#d6dade", edgecolor="white",
                    height=0.62, hatch="///", zorder=2)
            if rs >= 0.06:
                ax.text(left + rs / 2, yi, f"{100 * rs:.0f}%", ha="center",
                        va="center", fontsize=7, color="#555")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()                      # model on top
    ax.set_xlim(0, 1); ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[n]) for n in names]
    leg_names = list(names)
    if rows[0][2] > 0.01:                   # model residual present
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#d6dade", hatch="///"))
        leg_names.append("residual")
    ax.legend(handles, leg_names, fontsize=6.5, ncol=min(len(leg_names), 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.34)
    return _svg(fig)
