"""Statistical-equivalence report card from paired per-cell observables.

Reuses the workspace card library exactly as the existing equivalence
generators do (see scripts/pin_vecoli_equivalence_reference.py):
  - a `reference` dict declares axes {path: {label, group, criterion}}; for a
    ttest axis the criterion carries `ref_values` (the vEcoli per-cell list).
  - a `card` dict holds the MEASURED (v2ecoli) data at the same dotted paths;
    grade_card digs each. For a ttest axis the node is a _stat_node
    {values, mean, std, cv, n}.
  - grade_card(card, reference) -> {overall, axes}; verdict_json(...) -> v1 JSON;
    render_html(card, reference, ...) -> HTML.
"""
from __future__ import annotations

from typing import Any

from v2ecoli.library.report_card import (
    grade_card, verdict_json, render_html, _stat_node)

# (group, dotted path, label, observable-key). ttest criterion: within 5% pass,
# 5-10% drift, >10% & p<0.05 mismatch — matches the vEcoli equivalence cards.
CARD_AXES: list[dict[str, Any]] = [
    {"group": "Physiology", "path": "physiology.cell_mass",
     "label": "Cell mass", "key": "cell_mass"},
    {"group": "Physiology", "path": "physiology.growth_rate",
     "label": "Growth rate", "key": "growth_rate"},
]

_TTEST = {"type": "ttest", "within_pct": 0.05, "mismatch_pct": 0.10, "p_min": 0.05}


def build_report_card(left_by_cell: dict[str, list[float]],
                      right_by_cell: dict[str, list[float]], *,
                      model_ref: str = "",
                      reference_model: str = "vEcoli (fork)",
                      measured_model: str = "v2ecoli",
                      extra_axes: list[dict] | None = None
                      ) -> tuple[dict, str]:
    """Grade each axis (vEcoli=ref_values, v2ecoli=measured) -> (verdict_json, html).

    ``left_by_cell`` carries the REFERENCE (vEcoli) per-cell scalar lists and
    ``right_by_cell`` the MEASURED (v2ecoli) ones, keyed by observable name.
    """
    axes_defs = CARD_AXES + list(extra_axes or [])
    reference_axes: dict[str, Any] = {}
    card: dict[str, Any] = {}
    for spec in axes_defs:
        path, key = spec["path"], spec["key"]
        ref_vals = list(left_by_cell.get(key) or [])
        meas_vals = list(right_by_cell.get(key) or [])
        reference_axes[path] = {
            "label": spec["label"], "group": spec["group"],
            "criterion": {**_TTEST, "ref_values": ref_vals},
        }
        # set the measured stat node at the dotted path in `card`
        node = _stat_node(meas_vals)  # safe for [] (n=0 -> ungraded downstream)
        cur = card
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = node

    reference = {
        "title": "vEcoli ↔ v2ecoli statistical equivalence",
        "stimulus": {"reference_model": reference_model,
                     "measured_model": measured_model},
        "axes": reference_axes,
    }
    report = grade_card(card, reference)
    vjson = verdict_json(report, model_ref=model_ref,
                         reference_model=reference_model)
    html = render_html(card, reference, model_ref=model_ref)
    return vjson, html
