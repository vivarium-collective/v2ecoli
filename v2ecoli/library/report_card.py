"""Report-card grading + rendering (reference-driven, typed criteria).

The reference fixture declares **axes** — each with a presentation block
(group, label, units, how-it's-computed + analysis-script pointer, plot kind)
and a typed ``criterion`` (see ``card_criteria``). The measured *card* (an
ensemble analysis, optionally merged with omics/behavioral nodes) supplies the
value for each axis path. ``grade_card`` digs each path, grades it, and the
renderers lay it out grouped, with embedded inline-SVG plots and the verdict
band. One grader + one renderer serve both the organism-behavior card and the
v1<->v2 equivalence card (different reference sources, same machinery).

See ``docs/meta_report_cards.md``.
"""
from __future__ import annotations

import json
from typing import Any

from v2ecoli.library.card_criteria import grade_axis

_COLOR = {"within_tol": "#1a7f37", "drift": "#ef6c00",
          "mismatch": "#c62828", "ungraded": "#757575"}
_GLYPH = {"within_tol": "✓", "drift": "≈", "mismatch": "✗", "ungraded": "–"}
_RANK = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}
_DEFAULT_FOOTER = ("Behavioral report card — see docs/report_cards/README.md "
                   "for the index and how the cards compose.")


def dig(card: dict, path: str) -> Any:
    node = card
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def card_from_analysis(analysis: dict) -> dict:
    """Pull the single population_phenotype_basal result out of an analysis.json."""
    cards = (analysis.get("multiseed", {}) or {}).get("population_phenotype_basal", {}) or {}
    if not cards:
        raise KeyError("no multiseed.population_phenotype_basal in analysis")
    return next(iter(cards.values()))


def _set_path(card: dict, path: str, value: Any) -> None:
    node = card
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _stat_node(values: list[float]) -> dict:
    """Build a {values, mean, std, cv, n} node from per-cell values — the same
    shape the ttest/violin axes consume for scalar and KPI axes alike."""
    import statistics
    n = len(values)
    mean = sum(values) / n if n else 0.0
    std = statistics.pstdev(values) if n > 1 else 0.0
    cv = (std / abs(mean)) if mean else 0.0
    return {"values": list(values), "mean": mean, "std": std, "cv": cv, "n": n}


def merge_vectors(card: dict, reference: dict, sweep_dir: str,
                  generation_lower_bound: int = 0) -> dict:
    """Merge omics + exchange-flux nodes into a measured card by reading the
    sweep parquet (the array columns aren't in the scalar per-cell records).

    Omics nodes merge straight in (``omics.{transcriptome,proteome}.vector``).
    The exchange-flux ensemble-mean vector populates the scatter axis; named
    flux KPIs (declared via ``criterion.flux_id`` on a ttest axis) are sliced
    per-cell from the flux matrix using the ``flux_ids`` order pinned in the
    reference's scatter criterion. Heavy (~minute) — call at render time."""
    from v2ecoli.library.card_vectors import extract_vectors
    vec = extract_vectors(sweep_dir, generation_lower_bound)
    if vec.get("omics"):
        card.setdefault("omics", {}).update(vec["omics"])
    exch = (vec.get("fluxes") or {}).get("exchange")
    if not exch:
        return card
    per_cell = exch.get("per_cell") or []
    # per-flux measured std across cells (for scatter error bars + the table)
    if per_cell:
        n = len(per_cell)
        mean = exch["vector"]
        std = [(sum((row[j] - mean[j]) ** 2 for row in per_cell) / n) ** 0.5
               for j in range(len(mean))]
    else:
        std = [0.0] * len(exch["vector"])
    card.setdefault("fluxes", {})["exchange"] = {
        "vector": exch["vector"], "std": std, "n_cells": exch["n_cells"]}
    # flux_ids order is pinned on whichever axis declares the scatter criterion
    flux_ids = None
    for spec in (reference.get("axes") or {}).values():
        fid = spec.get("criterion", {}).get("flux_ids")
        if fid:
            flux_ids = fid
            break
    if not (flux_ids and per_cell):
        return card
    for path, spec in (reference.get("axes") or {}).items():
        kpi = spec.get("criterion", {}).get("flux_id")
        if kpi and kpi in flux_ids:
            j = flux_ids.index(kpi)
            _set_path(card, path, _stat_node([row[j] for row in per_cell]))
    return card


def grade_card(card: dict, reference: dict) -> dict:
    """Grade every axis the reference declares. Returns
    ``{overall, axes: {path: {...presentation, ...grade}}}``."""
    axes_out: dict[str, dict] = {}
    worst = "ungraded"
    for path, spec in (reference.get("axes") or {}).items():
        measured = dig(card, path)
        g = grade_axis(measured, spec.get("criterion", {}))
        axes_out[path] = {**spec, **g, "path": path, "measured": measured}
        if _RANK[g["verdict"]] > _RANK[worst]:
            worst = g["verdict"]
    return {"overall": worst, "axes": axes_out}


def _fmt_value(a: dict) -> str:
    """The headline value cell. For vector axes (r2 / flux_scatter) the graded
    value is a unitless R² — show it as such; for scalar axes it's a measured
    quantity with units and an optional display scale."""
    val = a.get("value")
    if val is None:
        return "—"
    ctype = a.get("criterion", {}).get("type")
    if ctype == "boolean":
        return ""  # the verdict badge + meter (pass/FAIL) carry it
    if ctype in ("r2", "flux_scatter"):
        return f"R² {val:.4g}"
    if isinstance(val, float):
        return f"{val * a.get('scale', 1.0):.4g} {a.get('units', '')}".strip()
    return str(val)


def _stationarity(measured: dict) -> tuple[bool, str]:
    """Read an axis's variance decomposition into (flagged, readout). Flagged =
    variance is generation-dominated AND monotonic — the ensemble is drifting
    across generations rather than fluctuating around a steady state. Non-
    grading: a diagnostic about ensemble quality, not a pass/fail of the axis."""
    var = measured.get("variance") if isinstance(measured, dict) else None
    if not var:
        return False, ""
    es, eg = var.get("eta_seed", 0.0), var.get("eta_gen", 0.0)
    rho, p = var.get("rho_gen"), var.get("p_gen")
    txt = f"seed {es:.0%} · gen {eg:.0%}"
    if rho is not None:
        txt += f" · ρ(gen)={rho:+.2f}"
    flagged = (eg > es and eg > 0.3 and rho is not None
               and abs(rho) > 0.4 and (p is None or p < 0.1))
    return flagged, txt


def _decompose(by_cell) -> dict | None:
    """eta_seed / eta_gen / rho_gen from ``[[seed, gen, value], ...]`` — computed
    at render time for the REFERENCE side (from ``criterion.ref_by_cell``) so the
    card can show both ensembles' variance structure side by side. Mirrors
    ``analysis._variance_decomposition`` (kept local to avoid a workflow import)."""
    if not by_cell or len(by_cell) < 6:
        return None
    s = [x[0] for x in by_cell]; g = [x[1] for x in by_cell]
    v = [float(x[2]) for x in by_cell]
    n = len(v); mu = sum(v) / n; sst = sum((x - mu) ** 2 for x in v)
    if sst == 0:
        return None

    def _gss(keys):
        ss = 0.0
        for k in set(keys):
            idx = [i for i, kk in enumerate(keys) if kk == k]
            gm = sum(v[i] for i in idx) / len(idx)
            ss += len(idx) * (gm - mu) ** 2
        return ss

    out = {"eta_seed": _gss(s) / sst, "eta_gen": _gss(g) / sst}
    try:
        from scipy.stats import spearmanr
        out["rho_gen"] = float(spearmanr(g, v).correlation)
    except Exception:
        pass
    return out


def _fmt_var(var: dict, name: str) -> str:
    t = f"{name}: seed {var.get('eta_seed', 0):.0%} · gen {var.get('eta_gen', 0):.0%}"
    if var.get("rho_gen") is not None:
        t += f" · ρ={var['rho_gen']:+.2f}"
    return t


def _variance_line(measured: dict, crit: dict, meas_label: str, ref_label: str) -> str:
    """Seed/gen variance breakdown for BOTH ensembles (measured from the card's
    variance node; reference computed from ``criterion.ref_by_cell``) so the
    reader can compare the structure — e.g. v1 seed-dominated vs v2 gen-dominated."""
    parts = []
    mvar = measured.get("variance") if isinstance(measured, dict) else None
    if mvar:
        parts.append(_fmt_var(mvar, meas_label))
    rvar = _decompose((crit or {}).get("ref_by_cell"))
    if rvar:
        parts.append(_fmt_var(rvar, ref_label))
    return " │ ".join(parts)


def _counts(report: dict) -> dict[str, int]:
    c = {v: 0 for v in _COLOR}
    for a in report["axes"].values():
        c[a["verdict"]] += 1
    return c


def _overall_label(report: dict) -> str:
    c = _counts(report)
    if c["mismatch"]:
        return "MISMATCH"
    if c["drift"]:
        return "DRIFT"
    if c["within_tol"]:
        return "PASS" + (" (partial)" if c["ungraded"] else "")
    return "UNGRADED"


# ---------------------------------------------------------------------------
# Markdown — compact text summary (no plots)
# ---------------------------------------------------------------------------

def render_markdown(card: dict, reference: dict, *, model_ref=None, generated=None) -> str:
    report = grade_card(card, reference)
    c = _counts(report)
    title = reference.get("title", "Basal-condition phenotype")
    stim = reference.get("stimulus", {})
    lines = [
        f"# {title} — report card", "",
        f"- **Model**: {model_ref or stim.get('blessed_model_ref') or '(unspecified)'}",
        f"- **Stimulus**: {stim.get('summary') or stim.get('ensemble') or stim.get('config') or stim.get('source') or ''}",
        f"- **Reference status**: {reference.get('status', 'unknown')}",
    ]
    if generated:
        lines.append(f"- **Generated**: {generated}")
    lines += ["", f"## Overall: {_overall_label(report)} "
              f"({c['within_tol']} ✓ · {c['drift']} ≈ · {c['mismatch']} ✗ · {c['ungraded']} –)", ""]
    sh = card.get("sim_health") if isinstance(card, dict) else None
    if sh and sh.get("n_total"):
        nt, nd, nf = sh["n_total"], sh.get("n_divided", 0), sh.get("n_failed", 0)
        ml = stim.get("measured_model") or "measured"
        lines += [(f"> **✓ Simulations ({ml}):** all {nd}/{nt} generations divided." if nf == 0
                   else f"> **⚠ Simulations ({ml}):** {nf} of {nt} generations hit the "
                        f"duration cap without dividing ({nd}/{nt} divided)."), ""]
    flagged = [(a.get("label", a["path"]), a["measured"].get("variance", {}))
               for a in report["axes"].values()
               if _stationarity(a.get("measured") or {})[0]]
    if flagged:
        lines += ["> **⚠ Stationarity:** " + ", ".join(
            f"`{n}` (gen {v.get('eta_gen', 0):.0%}, ρ={v.get('rho_gen', 0):+.2f})"
            for n, v in flagged)
            + " show generation-structured drift — the ensemble may not be at "
            "steady balanced growth across generations (burn-in insufficient or a "
            "generational instability). Diagnostic only; does not affect grades.", ""]
    # group axes
    groups: dict[str, list] = {}
    for a in report["axes"].values():
        groups.setdefault(a.get("group", "Other"), []).append(a)
    for gname, axes in groups.items():
        lines += [f"### {gname}", "",
                  "| Axis | Value | Criterion | Summary | Verdict |",
                  "|---|---|---|---|---|"]
        for a in axes:
            vals = _fmt_value(a)
            lines.append(f"| {a.get('label', a['path'])} | {vals} | {a['criterion_str']} "
                         f"| {a['meter']} | {_GLYPH[a['verdict']]} {a['verdict']} |")
        lines.append("")
    findings = reference.get("findings") or []
    if findings:
        lines += ["## Findings", ""] + [f"- {f}" for f in findings] + [""]
    lines += [f"_{reference.get('footer', _DEFAULT_FOOTER)}_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML — rich, grouped, plots embedded, click-to-expand
# ---------------------------------------------------------------------------

def _flux_table(measured: dict, crit: dict) -> str:
    """Companion table for the exchange-flux scatter: one row per *active*
    exchange (nonzero in reference or measured), with reference and measured
    shown as mean ± std (cell-to-cell). Sorted by |reference flux| so the
    metabolic majors (glucose, O2, NH4, CO2, …) lead. Appeared/disappeared
    exchanges are flagged."""
    ids = crit.get("flux_ids") or []
    rv = crit.get("ref_vector") or []
    rs = crit.get("ref_std") or [0.0] * len(rv)
    cv = measured.get("vector") or []
    cs = measured.get("std") or [0.0] * len(cv)
    eps = crit.get("active_eps", 1e-6)
    rows = []
    for i, mol in enumerate(ids):
        r, c = (rv[i] if i < len(rv) else 0.0), (cv[i] if i < len(cv) else 0.0)
        if abs(r) <= eps and abs(c) <= eps:
            continue
        flag = ("appeared" if abs(r) <= eps else
                "lost" if abs(c) <= eps else "")
        rsd = rs[i] if i < len(rs) else 0.0
        csd = cs[i] if i < len(cs) else 0.0
        ref_s = "—" if abs(r) <= eps else f"{r:+.3g} ± {rsd:.2g}"
        meas_s = "—" if abs(c) <= eps else f"{c:+.3g} ± {csd:.2g}"
        rows.append((abs(r), mol, ref_s, meas_s, flag))
    rows.sort(key=lambda t: t[0], reverse=True)
    trs = "".join(
        f"<tr class='{('flux-'+flag) if flag else ''}'>"
        f"<td class='fid'>{mol}{(' '+chr(0x2009)+'·'+chr(0x2009)+flag) if flag else ''}</td>"
        f"<td class='fnum'>{ref_s}</td><td class='fnum'>{meas_s}</td></tr>"
        for _, mol, ref_s, meas_s, flag in rows)
    return (f"<div class='ftbl'><table><thead><tr><th>Flux</th>"
            f"<th>Reference</th><th>Measured</th></tr></thead>"
            f"<tbody>{trs}</tbody></table>"
            f"<div class='ftnote'>mean ± std over n={measured.get('n_cells','?')} "
            f"cells · mmol/gDCW/h · neg=uptake, pos=secretion · "
            f"{len(rows)} active of {len(ids)}</div></div>")


def _gen_trend_svg(axis: dict, ref_label="reference", meas_label="measured") -> str:
    """Companion metric-vs-generation plot for a flagged generation-drift axis
    (or '' if no labeled points). Measured lineages (green) from the variance
    node's `by_cell`; reference lineages (grey) from `criterion.ref_by_cell`
    when the reference carries per-lineage labels."""
    measured = axis.get("measured") or {}
    var = measured.get("variance") or {}
    by_cell = var.get("by_cell")
    if not by_cell:
        return ""
    ref_by_cell = (axis.get("criterion") or {}).get("ref_by_cell")
    ref_rho = None
    if ref_by_cell:
        try:
            from scipy.stats import spearmanr
            ref_rho = float(spearmanr([c[1] for c in ref_by_cell],
                                      [c[2] for c in ref_by_cell]).correlation)
        except Exception:
            ref_rho = None
    try:
        from v2ecoli.library import card_plots
        return card_plots.generation_trend(
            by_cell, ref_by_cell, scale=axis.get("scale", 1.0),
            units=axis.get("units", ""), label=axis.get("label", ""),
            rho=var.get("rho_gen"), ref_rho=ref_rho,
            y_from_zero=axis.get("y_from_zero", False),
            ref_label=ref_label, meas_label=meas_label)
    except Exception as e:
        return f"<div class='ploterr'>trend plot unavailable: {type(e).__name__}</div>"


def _axis_plot_svg(axis: dict, ref_label="reference", meas_label="measured") -> str:
    """Render the axis's plot as inline SVG (or '' if none / no data)."""
    kind = axis.get("plot")
    measured = axis.get("measured") or {}
    crit = axis.get("criterion", {})
    try:
        from v2ecoli.library import card_plots
        if kind == "violin" and isinstance(measured, dict) and measured.get("values"):
            return card_plots.violin_strip(
                measured["values"], crit.get("ref_values"),
                label=axis.get("label", ""), units=axis.get("units", ""),
                scale=axis.get("scale", 1.0), y_from_zero=axis.get("y_from_zero", False),
                ref_label=ref_label, meas_label=meas_label)
        if kind == "loglog" and isinstance(measured, dict) and measured.get("vector"):
            return card_plots.loglog_scatter(
                measured["vector"], crit.get("ref_vector"),
                r2=axis.get("value"), label=axis.get("label", ""))
        if kind == "flux_scatter" and isinstance(measured, dict) and measured.get("vector"):
            svg = card_plots.flux_scatter(
                measured["vector"], crit.get("ref_vector"),
                ids=crit.get("flux_ids"), r2=axis.get("value"),
                active_eps=crit.get("active_eps", 1e-6),
                ref_std=crit.get("ref_std"), cand_std=measured.get("std"),
                label=axis.get("label", ""))
            return f"<div class='fluxwrap'>{svg}{_flux_table(measured, crit)}</div>"
    except Exception as e:  # a plot failure must not blank the report
        return f"<div class='ploterr'>plot unavailable: {type(e).__name__}</div>"
    return ""


def _sim_health_html(card: dict, label: str) -> str:
    """Run-quality banner: how the sims themselves went (divided vs hit the
    per-generation duration cap), separate from the graded phenotype axes."""
    sh = card.get("sim_health") if isinstance(card, dict) else None
    if not sh or not sh.get("n_total"):
        return ""
    nt, nd, nf = sh.get("n_total", 0), sh.get("n_divided", 0), sh.get("n_failed", 0)
    if nf == 0:
        return ("<section class='card simhealth ok'><div class='shbody'>✓ "
                f"<b>Simulations ({label}):</b> all {nd}/{nt} generations divided"
                "</div></section>")
    return ("<section class='card simhealth warn'><div class='shbody'>⚠ "
            f"<b>Simulations ({label}):</b> {nf} of {nt} generations hit the "
            f"duration cap without dividing ({nd}/{nt} divided)</div></section>")


def render_html(card: dict, reference: dict, *, model_ref=None, generated=None) -> str:
    report = grade_card(card, reference)
    c = _counts(report)
    title = reference.get("title", "Basal-condition phenotype")
    footer = reference.get("footer", _DEFAULT_FOOTER)
    stim = reference.get("stimulus", {})
    ref_label = stim.get("reference_model") or "reference"
    meas_label = stim.get("measured_model") or "measured"
    sim_html = _sim_health_html(card, meas_label)
    groups: dict[str, list] = {}
    for a in report["axes"].values():
        groups.setdefault(a.get("group", "Other"), []).append(a)

    def chip(v, n):
        return f"<span class='chip' style='background:{_COLOR[v]}'>{_GLYPH[v]} {n} {v.replace('_',' ')}</span>"

    sections = []
    flagged_axes = []  # axes whose variance is gen-dominated + monotonic
    for gname, axes in groups.items():
        gc = {v: sum(1 for a in axes if a["verdict"] == v) for v in _COLOR}
        rows = []
        for a in axes:
            vals = _fmt_value(a)
            sp = a.get("measured") if isinstance(a.get("measured"), dict) else {}
            spread = (f"±{sp['std'] * a.get('scale', 1.0):.3g} (CV {sp['cv']:.1%}, n={sp['n']})"
                      if "std" in sp and "cv" in sp else "")
            flagged, _ = _stationarity(sp)
            if flagged:
                flagged_axes.append(a.get("label", a["path"]))
            plot = _axis_plot_svg(a, ref_label, meas_label)
            trend = _gen_trend_svg(a, ref_label, meas_label) if flagged else ""
            summary = "plot + generation trend" if trend else "plot + detail"
            detail = (f"<details{' open' if flagged else ''}><summary>{summary}</summary>"
                      f"<div class='plotwrap'>{plot}{trend}</div></details>"
                      if (plot or trend) else "")
            vline = _variance_line(sp, a.get("criterion", {}), meas_label, ref_label)
            var_html = (f"<div class='var{' varflag' if flagged else ''}'>variance — "
                        f"{vline}{' ⚠ generation-drift' if flagged else ''}</div>"
                        if vline else "")
            rows.append(
                f"<tr class='verdict-{a['verdict']}'>"
                f"<td><div class='axhd'><span class='metric'>{a.get('label', a['path'])}</span>"
                f"<span class='badge' style='background:{_COLOR[a['verdict']]}'>{_GLYPH[a['verdict']]} {a['verdict'].replace('_',' ')}</span></div>"
                f"<div class='how'>{a.get('how','')}</div>"
                f"<div class='crit'><b>criterion:</b> {a['criterion_str']}</div>{var_html}{detail}</td>"
                f"<td class='val'>{vals}<div class='spread'>{spread}</div></td>"
                f"<td class='meter'>{a['meter']}</td></tr>")
        anchor = gname.lower().replace(" ", "-")
        sections.append(
            f"<section class='card' id='{anchor}'><div class='head'><h2>{gname}</h2>"
            f"<div class='chips'>{chip('within_tol', gc['within_tol'])}{chip('drift', gc['drift'])}"
            f"{chip('mismatch', gc['mismatch'])}{chip('ungraded', gc['ungraded'])}</div></div>"
            f"<table><thead><tr><th>axis</th><th>value</th><th>summary</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")

    nav = "".join(
        f"<a href='#{g.lower().replace(' ','-')}'>{g}</a>" for g in groups)
    stationarity_html = ("<section class='card stat-callout'><div class='head'>"
                         "<h2>⚠ Stationarity diagnostic</h2></div>"
                         "<div class='statbody'>"
                         f"<b>{len(flagged_axes)}</b> axis(es) show <b>generation-structured "
                         "drift</b> (variance gen-dominated <i>and</i> monotonic across "
                         "generations): " + ", ".join(f"<code>{x}</code>" for x in flagged_axes)
                         + ". The ensemble may not be at steady balanced growth across "
                         "generations — burn-in may be insufficient or a generational "
                         "instability. <i>Diagnostic only (does not affect grades).</i>"
                         "</div></section>") if flagged_axes else ""
    findings = reference.get("findings") or []
    findings_html = ("<section class='card'><div class='head'><h2>Findings</h2></div>"
                     "<ul class='findings'>" + "".join(f"<li>{f}</li>" for f in findings)
                     + "</ul></section>") if findings else ""
    stim = reference.get("stimulus", {})
    # one stimulus descriptor for the header, whatever shape the card is:
    # population cards carry `ensemble` (seeds × gens); others carry `summary`.
    stim_desc = stim.get("summary") or stim.get("ensemble") or stim.get("source") or ""
    sub_bits = [f"model <b>{model_ref or stim.get('blessed_model_ref') or '?'}</b>"]
    if stim_desc:
        sub_bits.append(stim_desc)
    sub_bits.append(f"reference {reference.get('status', '?')}")
    if generated:
        sub_bits.append(generated)
    subtitle = " · ".join(sub_bits)
    overall = _overall_label(report)
    ocolor = (_COLOR["mismatch"] if c["mismatch"] else _COLOR["drift"] if c["drift"]
              else _COLOR["within_tol"] if c["within_tol"] else _COLOR["ungraded"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — report card</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;}}
*{{box-sizing:border-box}} body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.45}}
header.top{{background:linear-gradient(135deg,#1f2937,#111827);color:#fff;padding:22px 28px}}
header.top h1{{margin:0;font-size:20px;font-weight:650}} header.top .sub{{margin-top:4px;color:#cbd5e1;font-size:13px}}
.obadge{{display:inline-block;margin-top:12px;padding:4px 12px;border-radius:999px;font-weight:700;background:{ocolor};color:#fff}}
nav.sticky{{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.95);backdrop-filter:saturate(1.4) blur(6px);border-bottom:1px solid var(--line);padding:10px 28px;display:flex;gap:8px;flex-wrap:wrap}}
nav.sticky a{{text-decoration:none;color:var(--ink);font-size:13px;padding:5px 10px;border-radius:999px;border:1px solid var(--line)}}
nav.sticky a:hover{{background:var(--bg)}}
main{{padding:24px 28px 64px;max-width:1100px;margin:0 auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 22px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card>.head{{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}}
.card>.head h2{{margin:0;font-size:16px;font-weight:620}}
.chips{{display:flex;gap:7px;flex-wrap:wrap}} .chip{{font-size:12px;font-weight:600;padding:3px 9px;border-radius:999px;color:#fff}}
table{{border-collapse:collapse;width:100%}} thead th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);padding:9px 18px;border-bottom:1px solid var(--line);background:#fafbfc}}
tbody td{{padding:11px 18px;border-bottom:1px solid #f1f3f5;vertical-align:top;font-size:13px}}
.axhd{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}} .metric{{font-weight:600}}
.badge{{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:6px;color:#fff;white-space:nowrap}}
.how{{color:var(--muted);font-size:11.5px;margin-top:4px;max-width:560px}} .crit{{font-size:12px;margin-top:4px;color:#374151}}
.val{{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:13px;color:#374151;white-space:nowrap}} .spread{{color:var(--muted);font-size:11px;margin-top:2px}}
.meter{{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);white-space:nowrap}}
.verdict-within_tol td:first-child{{box-shadow:inset 3px 0 0 {_COLOR['within_tol']}}} .verdict-drift td:first-child{{box-shadow:inset 3px 0 0 {_COLOR['drift']}}}
.verdict-mismatch td:first-child{{box-shadow:inset 3px 0 0 {_COLOR['mismatch']}}} .verdict-ungraded td:first-child{{box-shadow:inset 3px 0 0 {_COLOR['ungraded']}}}
details{{margin-top:8px}} summary{{cursor:pointer;font-size:12px;color:#1f6feb}} .plotwrap{{margin-top:8px}} .plotwrap svg{{max-width:100%;height:auto}}
.fluxwrap{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}} .fluxwrap svg{{flex:0 0 auto;max-width:420px}}
.ftbl{{flex:1 1 280px;min-width:260px}} .ftbl table{{width:auto}} .ftbl thead th{{padding:5px 10px;font-size:10px}}
.ftbl tbody td{{padding:4px 10px;border-bottom:1px solid #f1f3f5;font-size:11.5px}} .ftbl .fid{{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:11px}}
.ftbl .fnum{{font-family:"SF Mono",ui-monospace,Menlo,monospace;text-align:right;white-space:nowrap}}
.ftbl tr.flux-appeared{{background:#fdecea}} .ftbl tr.flux-appeared .fid{{color:{_COLOR['mismatch']};font-weight:600}}
.ftbl tr.flux-lost{{background:#fff3e0}} .ftbl tr.flux-lost .fid{{color:{_COLOR['drift']};font-weight:600}}
.ftnote{{color:var(--muted);font-size:10.5px;margin-top:6px;max-width:340px;line-height:1.35}}
.ploterr{{color:var(--muted);font-size:11px}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem;margin:0}} dt{{color:#cbd5e1}}
.var{{font-size:11px;color:var(--muted);margin-top:4px}} .var.varflag{{color:{_COLOR['drift']};font-weight:600}}
.simhealth .shbody{{padding:12px 18px;font-size:13.5px}}
.simhealth.ok{{border-left:4px solid {_COLOR['within_tol']}}} .simhealth.ok .shbody{{color:#14532d}}
.simhealth.warn{{border-left:4px solid {_COLOR['drift']};background:#fff8f0}} .simhealth.warn .shbody{{color:#7c4a03}}
.stat-callout{{border-left:4px solid {_COLOR['drift']}}} .stat-callout .statbody{{padding:12px 18px;font-size:13px;color:#374151;line-height:1.5}}
.statbody code{{background:#fff3e0;padding:1px 5px;border-radius:4px;font-size:12px}}
.findings{{margin:0;padding:14px 34px;color:#374151;font-size:13px}} footer{{color:var(--muted);font-size:12px;padding:0 28px 40px;max-width:1100px;margin:0 auto}}
</style></head><body>
<header class="top"><h1>{title} — report card</h1>
<div class="sub">{subtitle}</div>
<div class="obadge">{overall} &nbsp;·&nbsp; {c['within_tol']} ✓ &nbsp; {c['drift']} ≈ &nbsp; {c['mismatch']} ✗ &nbsp; {c['ungraded']} –</div></header>
<nav class="sticky">{nav}</nav>
<main>{sim_html}{stationarity_html}{''.join(sections)}{findings_html}</main>
<footer>{footer}</footer>
</body></html>"""


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
