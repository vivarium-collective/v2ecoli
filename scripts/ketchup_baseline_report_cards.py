#!/usr/bin/env python3
"""Behavioral report cards: v2ecoli baseline vs the two KETCHUP kinetic models.

Stimulus  : glucose-minimal aerobic growth (the basal condition).
Behavior  : central-carbon exchange/byproduct fluxes.
Reference : v2ecoli baseline FBA exchange fluxes (the pinned phenotype card,
            tests/fixtures/population_phenotype_basal_reference.json).
Candidate : KETCHUP k-ecoli74 / k-ecoli307 fitted exchange fluxes
            (from pbg-ketchup; /tmp/ketchup_exchange.json).

Both sides are normalized to glucose uptake = -100 so the report-card
`flux_scatter` criterion (identity-line R^2 on signed, matched fluxes) is
comparing relative byproduct partitioning rather than absolute units. Only the
metabolites both models exchange (the central-carbon core) are graded — KETCHUP
is a core-carbon model and does not exchange the amino acids / ions v2ecoli's
whole-cell model does, which is expected, not a phenotype change.
"""

import json
import os
import sys
import time
from pathlib import Path

WT = Path("/Users/eranagmon/code/v2e-ketchup-compare")
sys.path.insert(0, str(WT))
from v2ecoli.library.report_card import grade_card, render_html  # noqa: E402

OUT = WT / "docs" / "report_cards" / "ketchup_vs_baseline"
OUT.mkdir(parents=True, exist_ok=True)

SHARED = ["GLC[p]", "OXYGEN-MOLECULE[p]", "CARBON-DIOXIDE[p]", "ACET[p]",
          "AMMONIUM[c]", "SULFATE[p]"]
PRETTY = {"GLC[p]": "glucose", "OXYGEN-MOLECULE[p]": "O₂",
          "CARBON-DIOXIDE[p]": "CO₂", "ACET[p]": "acetate",
          "AMMONIUM[c]": "ammonium", "SULFATE[p]": "sulfate"}


def _normalize(d: dict) -> list[float]:
    """Scale a {flux_id: value} dict to glucose = -100."""
    g = d["GLC[p]"]
    scale = -100.0 / g if g else 1.0
    return [round(d[s] * scale, 4) for s in SHARED]


def main():
    # --- reference: v2ecoli baseline exchange fluxes (glucose-normalized) ----
    fixture = json.load(open(
        WT / "tests/fixtures/population_phenotype_basal_reference.json"))
    fc = fixture["axes"]["fluxes.exchange"]["criterion"]
    ids, vec = fc["flux_ids"], fc["ref_vector"]
    v2_raw = {s: vec[ids.index(s)] for s in SHARED}
    v2_norm = _normalize(v2_raw)

    # --- candidates: the two KETCHUP models (from pbg-ketchup) ---------------
    ket = json.load(open(OUT / "ketchup_exchange.json"))
    cand = {m: _normalize(ket[m]["exchange"]) for m in ("k-ecoli74", "k-ecoli307")}

    generated = time.strftime("%Y-%m-%d %H:%M")
    graded = {}
    for model in ("k-ecoli74", "k-ecoli307"):
        reference = {
            "title": f"KETCHUP {model} vs v2ecoli baseline — central-carbon exchange",
            "stimulus": {"reference_model": "v2ecoli baseline (FBA)",
                         "measured_model": f"KETCHUP {model}"},
            "footer": "Behavioral report card (PR #134 grader) · exchange fluxes "
                      "glucose-normalized to -100 · shared central-carbon metabolites only.",
            "axes": {
                "fluxes.exchange": {
                    "group": "Exchange fluxes",
                    "label": "Central-carbon exchange (glucose-normalized)",
                    "units": "flux per 100 glucose",
                    "how": "KETCHUP fitted exchange fluxes vs v2ecoli baseline FBA, "
                           "both normalized to glucose uptake = -100; signed "
                           "(uptake −, secretion +).",
                    "plot": "flux_scatter",
                    "criterion": {
                        "type": "flux_scatter",
                        "flux_ids": SHARED,
                        "ref_vector": v2_norm,
                        "active_eps": 1e-6,
                        "qual_eps": 1e-3,
                        # cross-MODEL comparison thresholds (not regression-tight):
                        "r2_min": 0.8,    # good agreement
                        "r2_drift": 0.4,  # moderate agreement
                    },
                },
            },
        }
        card = {"fluxes": {"exchange": {
            "vector": cand[model], "std": [0.0] * len(SHARED), "n_cells": 1}}}

        report = grade_card(card, reference)
        html = render_html(card, reference, model_ref=model, generated=generated)
        (OUT / f"{model}.html").write_text(html)
        axis = report["axes"]["fluxes.exchange"]
        graded[model] = {"verdict": report["overall"], "r2": axis.get("value"),
                         "meter": axis.get("meter"), "normalized": cand[model]}
        print(f"{model}: {report['overall']} | {axis.get('meter')}")

    # --- combined side-by-side index ----------------------------------------
    rows = []
    for i, s in enumerate(SHARED):
        rows.append(
            f"<tr><td>{PRETTY[s]} <code>{s}</code></td>"
            f"<td class='n'>{v2_norm[i]:+.1f}</td>"
            f"<td class='n'>{cand['k-ecoli74'][i]:+.1f}</td>"
            f"<td class='n'>{cand['k-ecoli307'][i]:+.1f}</td></tr>")
    cards_links = "".join(
        f"<a class='cardlink {graded[m]['verdict']}' href='{m}.html'>"
        f"<b>{m}</b><span>{graded[m]['meter']}</span></a>"
        for m in ("k-ecoli74", "k-ecoli307"))
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KETCHUP ↔ v2ecoli baseline — exchange-flux report cards</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  max-width:880px;margin:0 auto;padding:30px;color:#0f172a;background:#f8fafc}}
h1{{font-size:22px}} p{{color:#334155;line-height:1.55}}
table{{border-collapse:collapse;width:100%;margin:18px 0;background:#fff;
  border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #eef2f6;font-size:14px}}
th{{background:#f1f5f9;font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
.links{{display:flex;gap:14px;margin:18px 0}}
.cardlink{{flex:1;display:block;padding:14px 16px;border-radius:10px;text-decoration:none;
  color:#0f172a;border:1px solid #e2e8f0;background:#fff}}
.cardlink b{{display:block;font-size:15px}} .cardlink span{{font-size:12px;color:#475569}}
.cardlink.mismatch{{border-left:4px solid #dc2626}} .cardlink.drift{{border-left:4px solid #d97706}}
.cardlink.within_tol{{border-left:4px solid #059669}}
.note{{font-size:13px;color:#64748b;border-left:3px solid #cbd5e1;padding:6px 12px;background:#fff}}
</style></head><body>
<h1>KETCHUP kinetic models ↔ v2ecoli baseline</h1>
<p>Behavioral report cards (PR #134 grader) comparing the central-carbon
<b>exchange-flux phenotype</b> of two KETCHUP kinetic models against the v2ecoli
whole-cell baseline, under glucose-minimal aerobic growth. All fluxes are
glucose-normalized to −100 (uptake −, secretion +); the <code>flux_scatter</code>
criterion grades identity-line R² over the shared central-carbon exchanges.</p>
<div class="links">{cards_links}</div>
<table><thead><tr><th>exchange (per 100 glucose)</th><th>v2ecoli baseline</th>
<th>k-ecoli74</th><th>k-ecoli307</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note">Both KETCHUP core-metabolism models predict substantially more
respiration (O₂, CO₂) and acetate overflow per glucose than v2ecoli's baseline
FBA, which is near-fermentative with no overflow — the dominant disagreement.
KETCHUP fluxes are at the bounded fit (status maxIterations); v2ecoli values are
the 20-seed ensemble mean. The comparison is over shared central-carbon
exchanges only. Generated {generated}.</p>
</body></html>"""
    (OUT / "index.html").write_text(index)
    json.dump({"generated": generated, "shared": SHARED, "v2ecoli_norm": v2_norm,
               "graded": graded}, open(OUT / "comparison.json", "w"), indent=2)
    print(f"\nWrote {OUT}/index.html (+ per-model cards + comparison.json)")


if __name__ == "__main__":
    main()
