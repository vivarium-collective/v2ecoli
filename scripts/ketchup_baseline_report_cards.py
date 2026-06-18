#!/usr/bin/env python3
"""Multi-model behavioral report cards: v2ecoli baseline vs kinetic models.

Grades the v2ecoli whole-cell baseline central-carbon exchange phenotype against
external kinetic models — as report-card reference sources — using the PR #134
grader with MULTIPLE axes per card:

  * a whole-vector ``flux_scatter`` axis (identity-line R^2 + appeared/lost), and
  * per-metabolite ``rel_tol`` axes (O2, CO2, NH3) for a legible breakdown.

Exchange-comparison models (all glucose-normalized to -100; uptake −, secretion +):
  * v2ecoli baseline      — pinned population_phenotype_basal reference (MEASURED)
  * KETCHUP k-ecoli74     — pbg-ketchup KetchupEstimator (108-rxn core)
  * KETCHUP k-ecoli307    — pbg-ketchup KetchupEstimator (305-rxn)
  * Millard 2017          — millard2017_metabolism COPASI ODE (glucose + acetate)
  * v2ecoli + Millard
    FBA-bridge            — the ``millard_fba_bridge_harness`` composite: the
                            REAL whole-cell tFBA with the Millard 2017 ODE wired
                            in as a flux source pinning the central-carbon FBA
                            reactions. Its exchange vector is read live from
                            ``listeners.fba_results.external_exchange_fluxes``
                            (see scripts/run_ketchup_models.py).

Plus a non-exchange showcase figure:
  * KETCHUP FDH dynamic   — KetchupDynamicEstimator time-series fit of cell-free
                            FDH NADH(t) across 9 conditions (rendered as a
                            small-multiples data-vs-fit figure, not an exchange
                            card — a dynamic kinetic fit has no exchange vector).

Reference (graded-against) = v2ecoli baseline; each kinetic model is a candidate.

The per-model fitted fluxes / fit data are pre-computed by
``scripts/run_ketchup_models.py`` (real IPOPT runs via the pbg-ketchup env, and
a real WCM run for the FBA-bridge) and cached as JSON in OUT; this script only
grades + renders. Run that script first to refresh the underlying data.
"""

import json
import time
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from v2ecoli.library.report_card import grade_card, render_html  # noqa: E402

OUT = REPO / "docs" / "report_cards" / "ketchup_vs_baseline"
OUT.mkdir(parents=True, exist_ok=True)

SHARED = ["GLC[p]", "OXYGEN-MOLECULE[p]", "CARBON-DIOXIDE[p]", "ACET[p]",
          "AMMONIUM[c]", "SULFATE[p]"]
PRETTY = {"GLC[p]": "glucose", "OXYGEN-MOLECULE[p]": "O₂",
          "CARBON-DIOXIDE[p]": "CO₂", "ACET[p]": "acetate",
          "AMMONIUM[c]": "ammonium", "SULFATE[p]": "sulfate"}
# metabolites graded as individual rel_tol axes (non-zero v2ecoli reference)
PER_METAB = [("OXYGEN-MOLECULE[p]", "O₂"), ("CARBON-DIOXIDE[p]", "CO₂"),
             ("AMMONIUM[c]", "ammonium")]

# Exchange-comparison models, in display order.
EXCHANGE_MODELS = ["k-ecoli74", "k-ecoli307", "millard2017", "fba-bridge"]


def _normalize(d: dict, ids=SHARED) -> dict:
    """Scale a {flux_id: value} dict to glucose = -100; None for absent."""
    g = d.get("GLC[p]")
    scale = -100.0 / g if g else 1.0
    return {s: (round(d[s] * scale, 3) if d.get(s) is not None else None) for s in ids}


def _model_data():
    fixture = json.load(open(
        REPO / "tests/fixtures/population_phenotype_basal_reference.json"))
    fc = fixture["axes"]["fluxes.exchange"]["criterion"]
    ids, vec = fc["flux_ids"], fc["ref_vector"]
    v2 = _normalize({s: vec[ids.index(s)] for s in SHARED})

    ket = json.load(open(OUT / "ketchup_exchange.json"))
    k74 = _normalize(ket["k-ecoli74"]["exchange"])
    k307 = _normalize(ket["k-ecoli307"]["exchange"])

    # Millard 2017 central-carbon ODE: glucose uptake + acetate excretion only.
    millard_raw = json.load(open(OUT / "millard_exchange.json"))
    millard = _normalize({**{s: None for s in SHARED}, **millard_raw})

    # v2ecoli + Millard FBA-bridge harness: real WCM exchange vector (live from
    # listeners.fba_results.external_exchange_fluxes, time-averaged post burn-in).
    bridge_raw = json.load(open(OUT / "fba_bridge_exchange.json"))["shared"]
    bridge = _normalize(bridge_raw)

    cand = {"k-ecoli74": k74, "k-ecoli307": k307,
            "millard2017": millard, "fba-bridge": bridge}
    return v2, cand


ACCENT = {"k-ecoli74": "#2563eb", "k-ecoli307": "#059669",
          "millard2017": "#b45309", "fba-bridge": "#7c3aed"}
LABEL = {"k-ecoli74": "KETCHUP k-ecoli74", "k-ecoli307": "KETCHUP k-ecoli307",
         "millard2017": "Millard 2017 ODE",
         "fba-bridge": "v2ecoli + Millard FBA-bridge"}


def _reference_for(model_key, v2, cand):
    """A multi-axis reference: flux_scatter (whole vector) + per-metabolite rel_tol."""
    c = cand[model_key]
    # vectors restricted to metabolites BOTH models cover (for the scatter axis)
    ids = [s for s in SHARED if v2[s] is not None and c[s] is not None]
    ref_vec = [c[s] for s in ids]
    axes = {
        "fluxes.exchange": {
            "group": "Whole-vector agreement",
            "label": "Central-carbon exchange scatter (glucose-normalized)",
            "units": "flux per 100 glucose",
            "how": f"v2ecoli baseline exchange fluxes graded against {LABEL[model_key]} "
                   "as reference; identity-line R² over shared, glucose-normalized, "
                   "signed exchanges (+ appeared/lost).",
            "plot": "flux_scatter",
            "criterion": {"type": "flux_scatter", "flux_ids": ids, "ref_vector": ref_vec,
                          "active_eps": 1e-6, "qual_eps": 1e-3,
                          "r2_min": 0.8, "r2_drift": 0.4},
        },
    }
    for fid, nice in PER_METAB:
        if v2[fid] is None or c[fid] is None:
            continue
        axes[f"metab.{fid}"] = {
            "group": "Per-metabolite agreement (vs v2ecoli, glucose-normalized)",
            "label": f"{nice} flux",
            "units": "per 100 glucose",
            "how": f"v2ecoli {v2[fid]:+.0f} vs {LABEL[model_key]} {c[fid]:+.0f} "
                   "(reference); within 30% rel. tol.",
            "criterion": {"type": "rel_tol", "reference": c[fid], "tol_rel": 0.30},
        }
    return axes, ids


def _card_for(model_key, v2, ids):
    """Measured card = v2ecoli values, shaped for both axis kinds."""
    card = {"fluxes": {"exchange": {"vector": [v2[s] for s in ids],
                                    "std": [0.0] * len(ids), "n_cells": 1}},
            "metab": {}}
    for fid, _ in PER_METAB:
        if v2[fid] is not None:
            card["metab"][fid] = v2[fid]
    return card


def main():
    v2, cand = _model_data()
    generated = time.strftime("%Y-%m-%d %H:%M")
    graded = {}
    for mk in EXCHANGE_MODELS:
        axes, ids = _reference_for(mk, v2, cand)
        reference = {
            "title": f"v2ecoli baseline vs {LABEL[mk]} — central-carbon exchange",
            "stimulus": {"reference_model": LABEL[mk], "measured_model": "v2ecoli baseline (FBA)"},
            "footer": "Behavioral report card (PR #134 grader) · glucose-normalized · "
                      "multi-axis (whole-vector R² + per-metabolite rel-tol).",
            "axes": axes,
        }
        card = _card_for(mk, v2, ids)
        report = grade_card(card, reference)
        (OUT / f"{mk}.html").write_text(
            render_html(card, reference, model_ref=mk, generated=generated))
        scatter = report["axes"]["fluxes.exchange"]
        graded[mk] = {"overall": report["overall"], "r2": scatter.get("value"),
                      "meter": scatter.get("meter"), "n_matched": len(ids),
                      "axes": {p: a["verdict"] for p, a in report["axes"].items()}}
        print(f"{mk}: overall={report['overall']} | scatter {scatter.get('meter')}")

    _write_fdh_dynamic(generated)
    _write_index(v2, cand, graded, generated)
    json.dump({"generated": generated, "v2ecoli": v2, "candidates": cand,
               "graded": graded}, open(OUT / "comparison.json", "w"), indent=2)
    print(f"\nWrote {OUT}/index.html "
          f"(+ {len(EXCHANGE_MODELS)} exchange cards + fdh-dynamic.html + comparison.json)")
    return graded


# ---------------------------------------------------------------------------
# FDH dynamic NADH(t) showcase figure (not an exchange card)
# ---------------------------------------------------------------------------

def _svg_panel(title, t_fit, y_fit, t_dat, y_dat, w=240, h=150,
               accent="#0ea5e9"):
    """One small-multiple panel: fitted NADH curve + measured points."""
    pad_l, pad_b, pad_t, pad_r = 30, 22, 16, 8
    allt = (list(t_fit) + list(t_dat)) or [0, 1]
    ally = (list(y_fit) + list(y_dat)) or [0, 1]
    xlo, xhi = min(allt), max(allt) or 1.0
    ylo, yhi = min(ally + [0.0]), max(ally) or 1.0
    xr = (xhi - xlo) or 1.0
    yr = (yhi - ylo) or 1.0

    def X(x):
        return pad_l + (w - pad_l - pad_r) * (x - xlo) / xr

    def Y(y):
        return h - pad_b - (h - pad_b - pad_t) * (y - ylo) / yr

    line = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(t_fit, y_fit))
    dots = "".join(
        f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.2" '
        f'fill="#0f172a" fill-opacity="0.65"/>'
        for x, y in zip(t_dat, y_dat))
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'style="background:#fff;border:1px solid #e2e8f0;border-radius:8px">'
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" '
        f'stroke="#cbd5e1"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" '
        f'stroke="#cbd5e1"/>'
        f'<polyline fill="none" stroke="{accent}" stroke-width="2" '
        f'points="{line}"/>{dots}'
        f'<text x="{pad_l+2}" y="{pad_t+2}" font-size="11" '
        f'fill="#334155" font-weight="600">{title}</text>'
        f'<text x="{(w)/2:.0f}" y="{h-4}" font-size="9" fill="#94a3b8" '
        f'text-anchor="middle">time (s)</text>'
        f'</svg>')


def _write_fdh_dynamic(generated):
    data = json.load(open(OUT / "fdh_dynamic.json"))
    exps = list(data["nadh_fit"].keys())
    panels = "".join(
        _svg_panel(e, data["nadh_time"][e], data["nadh_fit"][e],
                   data["data_time"][e], data["data_nadh"][e])
        for e in exps)
    sse = data.get("sse")
    status = data.get("status", "—")
    npar = data.get("n_parameters")
    nexp = data.get("n_experiments", len(exps))
    solve = data.get("solve_time")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KETCHUP FDH dynamic fit — NADH(t)</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  max-width:920px;margin:0 auto;padding:26px;color:#0f172a;background:#f8fafc}}
h1{{font-size:20px}} p{{color:#334155;line-height:1.55;font-size:14px}}
.kpis{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.kpi{{background:#fff;border:1px solid #e2e8f0;border-left:4px solid #0ea5e9;
  border-radius:8px;padding:8px 12px;font-size:13px}}
.kpi b{{display:block;font-size:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:10px;margin-top:14px}}
.legend{{font-size:12px;color:#475569;margin-top:8px}}
.legend .ln{{display:inline-block;width:18px;height:3px;background:#0ea5e9;
  vertical-align:middle;margin-right:4px}}
.legend .dt{{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:#0f172a;opacity:.65;vertical-align:middle;margin:0 4px 0 12px}}
.note{{font-size:12px;color:#64748b;margin-top:14px}}
</style></head><body>
<h1>KETCHUP FDH — dynamic (time-series) kinetic fit</h1>
<p>The <code>KetchupDynamicEstimator</code> (pbg-ketchup) fits enzyme-kinetic
parameters of the cell-free formate-dehydrogenase (FDH) mechanism to measured
NADH(t) progress curves across {nexp} experimental conditions, via real
IPOPT optimization. This is a <b>dynamic kinetic fit</b>, not a steady-state
exchange model — it has no central-carbon exchange vector, so it is shown here
as its own data-vs-fit figure rather than graded on the exchange scatter.</p>
<div class="kpis">
  <span class="kpi"><b>{status}</b>IPOPT status</span>
  <span class="kpi"><b>{sse:.3g}</b>SSE (fit residual)</span>
  <span class="kpi"><b>{nexp}</b>conditions (A1–A{nexp})</span>
  <span class="kpi"><b>{npar}</b>fitted parameters</span>
  <span class="kpi"><b>{solve:.2g}s</b>solve time</span>
</div>
<div class="legend"><span class="ln"></span>fitted NADH(t)
  <span class="dt"></span>measured NADH</div>
<div class="grid">{panels}</div>
<p class="note">Real KETCHUP/IPOPT fit (pbg-ketchup KetchupDynamicEstimator,
model FDH). Each panel is one condition: line = fitted trajectory, dots =
measured NADH. Generated {generated}.</p>
</body></html>"""
    (OUT / "fdh-dynamic.html").write_text(doc)
    print(f"fdh-dynamic: status={status} sse={sse:.3g} nexp={nexp}")


def _write_index(v2, cand, graded, generated):
    order = EXCHANGE_MODELS
    rows = []
    for s in SHARED:
        cells = "".join(
            f"<td class='n'>{cand[m][s]:+.0f}</td>" if cand[m][s] is not None
            else "<td class='na'>—</td>" for m in order)
        rows.append(f"<tr><td>{PRETTY[s]} <code>{s}</code></td>"
                    f"<td class='n ref'>{v2[s]:+.0f}</td>{cells}</tr>")
    links = "".join(
        f"<a class='cardlink {graded[m]['overall']}' href='{m}.html'>"
        f"<b>{LABEL[m]}</b><span>{graded[m]['meter']}</span>"
        f"<span class='ov'>overall: {graded[m]['overall'].replace('_',' ')}</span></a>"
        for m in order)
    GLY = {"within_tol": "✓ matches", "drift": "~ drifts", "mismatch": "✗ diverges",
           "ungraded": "– n/a"}
    bridge = cand["fba-bridge"]
    results = (
        f"<b>v2ecoli ↔ Millard 2017 agree</b> on the overflow axis: both run "
        f"glucose-limited with <b>no acetate secretion</b> "
        f"({GLY.get(graded['millard2017']['overall'])} — but Millard only exchanges "
        f"glucose + acetate, so the match is over 2 metabolites). "
        f"<b>Both KETCHUP core models diverge</b> "
        f"({GLY.get(graded['k-ecoli74']['overall'])}): per 100 glucose they respire "
        f"far harder (O₂ ≈ −135 vs v2ecoli −9; CO₂ ≈ +145 vs +33) and secrete "
        f"acetate (≈ +71, absent in v2ecoli → 'appeared'/'lost'). "
        f"<b>The v2ecoli + Millard FBA-bridge</b> "
        f"({GLY.get(graded['fba-bridge']['overall'])}) sits in between on respiration "
        f"(O₂ ≈ {bridge['OXYGEN-MOLECULE[p]']:+.0f} per 100 glucose — ~3× the bare "
        f"baseline’s −9, but still far below the KETCHUP cores’ −135) while keeping "
        f"the baseline’s glucose-limited, acetate-free, CO₂-quiet exchange signature. "
        f"So pinning v2ecoli’s central-carbon FBA to the Millard ODE nudges it toward "
        f"the kinetic cores’ respiration without flipping it into acetate overflow.")
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v2ecoli baseline vs kinetic models — exchange-flux report cards</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  max-width:960px;margin:0 auto;padding:30px;color:#0f172a;background:#f8fafc}}
h1{{font-size:22px}} h2{{font-size:15px;text-transform:uppercase;letter-spacing:.04em;
  color:#475569;border-bottom:2px solid #cbd5e1;padding-bottom:5px;margin-top:30px}}
p{{color:#334155;line-height:1.55}}
table{{border-collapse:collapse;width:100%;margin:14px 0;background:#fff;
  border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #eef2f6;font-size:13px}}
th{{background:#f1f5f9;font-size:11px;text-transform:uppercase;letter-spacing:.03em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} td.ref{{font-weight:650;background:#f8fafc}}
td.na{{text-align:right;color:#cbd5e1}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
.links{{display:flex;gap:12px;margin:14px 0;flex-wrap:wrap}}
.cardlink{{flex:1;min-width:210px;display:block;padding:14px 16px;border-radius:10px;
  text-decoration:none;color:#0f172a;border:1px solid #e2e8f0;background:#fff}}
.cardlink b{{display:block;font-size:14px}} .cardlink span{{display:block;font-size:12px;color:#475569}}
.cardlink .ov{{margin-top:4px;font-weight:600}}
.cardlink.mismatch{{border-left:4px solid #dc2626}} .cardlink.drift{{border-left:4px solid #d97706}}
.cardlink.within_tol{{border-left:4px solid #059669}} .cardlink.ungraded{{border-left:4px solid #94a3b8}}
.cardlink.figure{{border-left:4px solid #0ea5e9}}
.results{{background:#fff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;
  border-radius:10px;padding:16px 18px;font-size:14px;line-height:1.6}}
.note{{font-size:13px;color:#64748b;margin-top:10px}}
</style></head><body>
<h1>v2ecoli baseline ↔ kinetic models: central-carbon exchange</h1>
<p>Behavioral report cards (PR #134 grader) grading the v2ecoli whole-cell
baseline exchange phenotype against kinetic models as reference sources — two
KETCHUP models (<a href="https://github.com/vivarium-collective/pbg-ketchup">pbg-ketchup</a>),
the Millard 2017 central-carbon ODE, and the
<code>millard_fba_bridge_harness</code> composite (the real whole-cell tFBA with
the Millard ODE pinning its central-carbon reactions). Each card is multi-axis:
a whole-vector <code>flux_scatter</code> R² plus per-metabolite rel-tol axes.
The KETCHUP <b>FDH dynamic</b> fit is shown as a separate NADH(t) figure.</p>

<h2>Results</h2>
<div class="results">{results}</div>

<h2>Report cards</h2>
<div class="links">{links}
<a class="cardlink figure" href="fdh-dynamic.html"><b>KETCHUP FDH dynamic</b>
<span>NADH(t) time-series fit · 9 conditions</span>
<span class="ov">figure: data vs fit</span></a></div>

<h2>Exchange fluxes (per 100 glucose)</h2>
<table><thead><tr><th>exchange</th><th>v2ecoli (ref)</th>
<th>k-ecoli74</th><th>k-ecoli307</th><th>Millard 2017</th>
<th>FBA-bridge</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">Uptake −, secretion +. Millard 2017 only exchanges glucose +
acetate (—  = not represented). KETCHUP fluxes at the bounded fit; v2ecoli is the
20-seed population_phenotype_basal ensemble mean; Millard at ODE steady state;
the FBA-bridge vector is the time-averaged
<code>listeners.fba_results.external_exchange_fluxes</code> from a live
<code>millard_fba_bridge_harness</code> run off the ParCa cache (real WCM, viable
over the window). Generated {generated}.</p>
</body></html>"""
    (OUT / "index.html").write_text(index)


if __name__ == "__main__":
    main()
