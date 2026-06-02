"""Report-card grading + rendering for the meta-tier basal-phenotype card.

One implementation of the grade logic, shared by the grade test
(``tests/test_basal_phenotype_card.py``) and the report renderer
(``reports/basal_phenotype_card_report.py``). See ``docs/meta_report_cards.md``
for the card's design.

A *card* is the measurement emitted by ``BasalPhenotypeCard.analyze`` (under
``results["multiseed"]["basal_phenotype_card"][<group>]``). A *reference* is
the pinned blessed-run values + tolerances in
``tests/fixtures/basal_phenotype_reference.json``. Grading compares the two.
"""
from __future__ import annotations

import json
from typing import Any


# Human labels for the gradeable paths into a card dict.
GRADE_LABELS = {
    "growth.doubling_time.mean": "Growth — doubling time (s)",
    "composition.protein_fraction.mean": "Composition — protein / dry weight",
    "composition.rna_fraction.mean": "Composition — RNA / dry weight",
    "composition.dna_fraction.mean": "Composition — DNA / dry weight",
}


def dig(card: dict, path: str) -> Any:
    """Resolve a dotted path (e.g. ``growth.doubling_time.mean``) into a card."""
    node = card
    for part in path.split("."):
        node = node[part]
    return node


def card_from_analysis(analysis: dict) -> dict:
    """Pull the single basal_phenotype_card result out of an analysis.json dict.

    The runner nests results as ``{scale: {name: {group_key: result}}}``. A
    basal ensemble has one variant group, so we return that one card.
    """
    cards = (analysis.get("multiseed", {}) or {}).get("basal_phenotype_card", {}) or {}
    if not cards:
        raise KeyError("no multiseed.basal_phenotype_card in analysis")
    return next(iter(cards.values()))


def grade_basal_phenotype(card: dict, reference: dict) -> dict:
    """Grade a measured card against a pinned reference.

    ``reference['grades']`` maps a dotted card path to
    ``{"reference": <value|None>, "tol_rel": <float>}``. A grade passes if the
    measured value is within ``tol_rel`` (relative) of a non-null reference. A
    null reference is *ungraded* (reference not yet pinned), not a failure.

    Returns ``{"overall", "grades": {path: {...}}}`` where overall is one of
    ``"pass" | "fail" | "ungraded"``.
    """
    grades: dict[str, dict] = {}
    any_graded = False
    all_pass = True
    for path, spec in reference.get("grades", {}).items():
        ref = spec.get("reference")
        tol = spec.get("tol_rel")
        try:
            got = dig(card, path)
        except (KeyError, TypeError):
            got = None
        if ref is None or got is None:
            grades[path] = {"reference": ref, "measured": got, "tol_rel": tol,
                            "verdict": "ungraded"}
            continue
        any_graded = True
        ok = ref != 0 and abs(got - ref) <= abs(tol * ref)
        all_pass = all_pass and ok
        grades[path] = {"reference": ref, "measured": got, "tol_rel": tol,
                        "verdict": "pass" if ok else "fail"}
    overall = "ungraded" if not any_graded else ("pass" if all_pass else "fail")
    return {"overall": overall, "grades": grades}


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return "—" if x is None else str(x)


def render_markdown(card: dict, reference: dict, *, model_ref: str | None = None,
                    generated: str | None = None) -> str:
    """Render the report card as Markdown."""
    report = grade_basal_phenotype(card, reference)
    badge = {"pass": "PASS ✅", "fail": "FAIL ❌", "ungraded": "UNGRADED (reference pending)"}
    lines = [
        "# Basal-condition phenotype — report card",
        "",
        f"- **Model**: {model_ref or reference.get('stimulus', {}).get('blessed_model_ref') or '(unspecified)'}",
        f"- **Stimulus**: `{reference.get('stimulus', {}).get('config', 'configs/basal_phenotype_card.json')}`",
        f"- **Ensemble**: {card.get('n_cells', 0)} cells after burn-in "
        f"(generation_lower_bound={card.get('generation_lower_bound', 0)})",
        f"- **Reference status**: {reference.get('status', 'unknown')}",
    ]
    if generated:
        lines.append(f"- **Generated**: {generated}")
    lines += ["", f"## Overall: {badge[report['overall']]}", "",
              "| Axis | Measured (mean) | spread | Reference | Tol | Verdict |",
              "|---|---|---|---|---|---|"]
    # spread (std / cv) for each measured axis lives next to its mean in the card
    spread_of = {
        "growth.doubling_time.mean": ("growth.doubling_time.std", "growth.doubling_time.cv"),
        "composition.protein_fraction.mean": ("composition.protein_fraction.std", "composition.protein_fraction.cv"),
        "composition.rna_fraction.mean": ("composition.rna_fraction.std", "composition.rna_fraction.cv"),
        "composition.dna_fraction.mean": ("composition.dna_fraction.std", "composition.dna_fraction.cv"),
    }
    vbadge = {"pass": "✅ pass", "fail": "❌ FAIL", "ungraded": "— ungraded"}
    for path, g in report["grades"].items():
        label = GRADE_LABELS.get(path, path)
        try:
            std = dig(card, spread_of[path][0]); cv = dig(card, spread_of[path][1])
            spread = f"±{_fmt(std)} (CV {_fmt(cv)})"
        except (KeyError, TypeError):
            spread = "—"
        tol = f"±{g['tol_rel']:.0%}" if isinstance(g.get("tol_rel"), (int, float)) else "—"
        lines.append(f"| {label} | {_fmt(g['measured'])} | {spread} | "
                     f"{_fmt(g['reference'])} | {tol} | {vbadge[g['verdict']]} |")
    lines += ["", "_Meta-tier card. A failure blocks merge; grades only move up. "
              "See `docs/meta_report_cards.md`._", ""]
    return "\n".join(lines)


def render_html(card: dict, reference: dict, *, model_ref: str | None = None,
                generated: str | None = None) -> str:
    """Render the report card as a standalone HTML page."""
    report = grade_basal_phenotype(card, reference)
    color = {"pass": "#1a7f37", "fail": "#cf222e", "ungraded": "#9a6700"}
    label_txt = {"pass": "PASS", "fail": "FAIL", "ungraded": "UNGRADED"}
    spread_of = {
        "growth.doubling_time.mean": ("growth.doubling_time.std", "growth.doubling_time.cv"),
        "composition.protein_fraction.mean": ("composition.protein_fraction.std", "composition.protein_fraction.cv"),
        "composition.rna_fraction.mean": ("composition.rna_fraction.std", "composition.rna_fraction.cv"),
        "composition.dna_fraction.mean": ("composition.dna_fraction.std", "composition.dna_fraction.cv"),
    }
    rows = []
    for path, g in report["grades"].items():
        try:
            std = dig(card, spread_of[path][0]); cv = dig(card, spread_of[path][1])
            spread = f"&plusmn;{_fmt(std)} (CV {_fmt(cv)})"
        except (KeyError, TypeError):
            spread = "&mdash;"
        tol = f"&plusmn;{g['tol_rel']:.0%}" if isinstance(g.get("tol_rel"), (int, float)) else "&mdash;"
        rows.append(
            f"<tr><td>{GRADE_LABELS.get(path, path)}</td><td>{_fmt(g['measured'])}</td>"
            f"<td>{spread}</td><td>{_fmt(g['reference'])}</td><td>{tol}</td>"
            f"<td style='color:{color[g['verdict']]};font-weight:600'>{g['verdict']}</td></tr>")
    o = report["overall"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Basal-phenotype report card</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.4rem}} .badge{{display:inline-block;padding:.2rem .7rem;border-radius:.5rem;color:#fff;font-weight:700;background:{color[o]}}}
 table{{border-collapse:collapse;width:100%;margin-top:1rem}} th,td{{border:1px solid #d0d7de;padding:.5rem .6rem;text-align:left}}
 th{{background:#f6f8fa}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem}} dt{{color:#656d76}}
 footer{{margin-top:1.5rem;color:#656d76;font-size:.9rem}}
</style></head><body>
<h1>Basal-condition phenotype &mdash; report card</h1>
<p class="badge">{label_txt[o]}</p>
<dl>
 <dt>Model</dt><dd>{model_ref or reference.get('stimulus', {}).get('blessed_model_ref') or '(unspecified)'}</dd>
 <dt>Stimulus</dt><dd><code>{reference.get('stimulus', {}).get('config', 'configs/basal_phenotype_card.json')}</code></dd>
 <dt>Ensemble</dt><dd>{card.get('n_cells', 0)} cells after burn-in (generation_lower_bound={card.get('generation_lower_bound', 0)})</dd>
 <dt>Reference status</dt><dd>{reference.get('status', 'unknown')}</dd>
 {f'<dt>Generated</dt><dd>{generated}</dd>' if generated else ''}
</dl>
<table><thead><tr><th>Axis</th><th>Measured (mean)</th><th>Spread</th><th>Reference</th><th>Tol</th><th>Verdict</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<footer>Meta-tier card. A failure blocks merge; grades only move up.
See <code>docs/meta_report_cards.md</code>.</footer>
</body></html>"""


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
