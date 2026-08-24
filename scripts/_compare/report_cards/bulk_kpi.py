"""Generic bulk-molecule KPI comparison card.

Grades a study's config-specific BULK observables — candidate (v2ecoli) vs
reference (vEcoli) — for whatever bulk molecule ids the study declared via
``observable_bulk_ids`` (emitted on both arms under
``listeners.observable_bulk.<id>``). One card serves every config: the violacein
intracellular titer (``VIOLACEIN[c]``), antibiotic drug-target complexes
(``mecillinam[p]-EG10606-MONOMER[i]``), free target, uptake — the card just reads
whatever was declared.

Grading is the candidate/reference relative delta on the final value (the same
unit-robust scheme the violacein card uses — both arms divide through the same
number, so it is correct whatever native unit the count carries). A declared id
that neither arm emitted degrades to ungraded, named, so the gap is visible.
"""

from __future__ import annotations

import html as _html
import os

from process_bigraph.composite import as_step

from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.verdict import worst

WITHIN = 0.05   # |rel Δ| ≤ 5% → within_tol (both arms implement the same model)
DRIFT = 0.10    # ≤ 10% → drift; above → mismatch


def _grade_rel(got, ref) -> str:
    if got is None or ref is None:
        return "ungraded"
    if ref == 0:
        return "within_tol" if got == 0 else "mismatch"
    rel = abs(got - ref) / abs(ref)
    return "within_tol" if rel <= WITHIN else ("drift" if rel <= DRIFT else "mismatch")


def _final(trace):
    """Last value of a (times, values) trace, or None."""
    if not trace:
        return None
    try:
        _t, vals = trace
        return float(vals[-1]) if len(vals) else None
    except Exception:  # noqa: BLE001
        return None


def _read_seed(dir_path: str, prefix: str, seed: int, leaves) -> dict:
    from scripts.compare_matched_trajectories import read_pbg_local
    path = os.path.join(dir_path or "", f"{prefix}_seed{seed:02d}.zarr")
    if not dir_path or not os.path.isdir(path):
        return {}
    try:
        return read_pbg_local(path, leaves)
    except Exception:  # noqa: BLE001
        return {}


def _mean_final(state, dir_key, prefix, leaf):
    """Mean over seeds of the final value of `leaf` on one arm; None if absent."""
    n = max(int(state.get("seeds") or 1), 1)
    vals = []
    for seed in range(n):
        d = _read_seed(state.get(dir_key), prefix, seed, [leaf])
        f = _final(d.get(leaf))
        if f is not None:
            vals.append(f)
    return (sum(vals) / len(vals)) if vals else None


def _declared_ids(state) -> list:
    """Bulk ids the study declared. Explicit state key first, else the study
    config's observable_bulk_ids."""
    ids = state.get("observable_bulk_ids")
    if ids:
        return list(ids)
    cfg = state.get("config") or {}
    return list(cfg.get("observable_bulk_ids") or [])


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="bulk_kpi_report_card",
         aliases=["bulk_kpi"])
def update_bulk_kpi_report_card(state):
    ids = _declared_ids(state)
    if not ids:
        return {"card_html": (
            '<p style="color:#6b7280">No bulk KPIs declared for this study — add '
            '<code>observable_bulk_ids</code> (e.g. <code>VIOLACEIN[c]</code>) to '
            'grade config-specific bulk molecules candidate-vs-reference.</p>'),
            "verdict": "ungraded", "axes": []}

    axes, rows = [], []
    for mol in ids:
        ve = _mean_final(state, "ve_dir", "vecoli", mol)      # reference
        v2 = _mean_final(state, "v2_dir", "v2ecoli", mol)     # candidate
        verdict = _grade_rel(v2, ve)
        rel = (None if (v2 is None or ve is None or ve == 0)
               else (v2 - ve) / abs(ve))
        meter = "—" if rel is None else f"Δ = {rel * 100:+.1f}%"
        axes.append({
            "id": f"bulk.{mol}", "label": mol, "verdict": verdict,
            "value": v2, "meter": meter,
            "detail": {"candidate": v2, "reference": ve, "delta_rel": rel}})
        fmt = lambda x: "—" if x is None else (f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}")
        rows.append(
            f'<tr><td style="padding:2px 10px"><code>{_html.escape(mol)}</code></td>'
            f'<td style="padding:2px 10px;text-align:right">{fmt(ve)}</td>'
            f'<td style="padding:2px 10px;text-align:right">{fmt(v2)}</td>'
            f'<td style="padding:2px 10px;text-align:right">{_html.escape(meter)}</td>'
            f'<td style="padding:2px 10px">{_html.escape(verdict)}</td></tr>')

    html = (
        '<p style="color:#374151;font-size:13px">Config-specific bulk KPIs — final '
        'count per arm (mean over seeds), graded on the candidate/reference '
        'relative delta.</p>'
        '<table style="border-collapse:collapse;font-size:13px">'
        '<thead><tr style="text-align:left">'
        '<th style="padding:2px 10px">bulk molecule</th><th>vEcoli</th>'
        '<th>v2ecoli</th><th>Δ</th><th>verdict</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>")
    return {"card_html": html, "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["bulk_kpi_report_card"] = update_bulk_kpi_report_card
