"""Per-condition report_card_verdict.json emission for the comparison harness.

Maps the unification model onto the existing report_card_axis evaluator:
ONE verdict JSON per CONDITION; each report CARD is a GROUP; a graded card's
internal axes are the group's axes (worst-of = the card's overall verdict).
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    # The substrate's single-source atomic write (write-to-.tmp then
    # os.replace) -- reused so a concurrent reader (e.g. comparison_matrix's
    # disk loader, or the dashboard) never observes a half-written
    # report_card_verdict.json. Optional import: this module must still work
    # hermetically when the substrate isn't on PYTHONPATH (e.g. a unit test
    # importing only scripts/_compare), falling back to a plain write below
    # (see v2ecoli/workflow/comparison_materialize.py's ``_write_text`` for
    # the same pattern).
    from vivarium_workbench.lib.atomic_io import atomic_write_text
except ImportError:  # pragma: no cover - exercised when substrate absent
    atomic_write_text = None

# Severity order matches pbg_v2ecoli/evaluators.py::_SEVERITY and the evaluator's
# worst-of-axes aggregation.
_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def worst(verdicts) -> str:
    """The most severe verdict in an iterable; 'ungraded' if empty/all unknown."""
    vs = [v for v in verdicts if v in _SEVERITY]
    return max(vs, key=lambda v: _SEVERITY[v]) if vs else "ungraded"


def build_condition_verdict(condition: str, card_verdicts: dict) -> dict:
    """Assemble the report_card_verdict/v1 doc for one condition.

    card_verdicts maps card_name -> {"verdict": str, "axes": list[dict]}. Cards
    with no axes (config/parca) become an 'ungraded' group. Top-level 'overall'
    is the worst across all groups.
    """
    groups: dict[str, dict] = {}
    for card_name, cv in card_verdicts.items():
        axes = (cv or {}).get("axes") or []
        gverdict = (cv or {}).get("verdict") or worst(
            a.get("verdict", "ungraded") for a in axes)
        groups[card_name] = {"verdict": gverdict, "axes": axes}
    overall = worst(g["verdict"] for g in groups.values())
    return {
        "schema": "report_card_verdict/v1",
        "model_ref": f"v2ecoli @ {condition}",
        "reference_model": f"vEcoli @ {condition}",
        "generated": "",
        "overall": overall,
        "groups": groups,
    }


def write_condition_verdict(card_root, condition: str, card_verdicts: dict) -> Path:
    """Write <card_root>/<condition>/report_card_verdict.json; return its path."""
    out = Path(card_root) / condition
    out.mkdir(parents=True, exist_ok=True)
    doc = build_condition_verdict(condition, card_verdicts)
    path = out / "report_card_verdict.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_study_verdict(study_dir, verdict: dict) -> Path:
    """Write ``verdict`` (a ``report_card_verdict/v1`` dict) to
    ``<study_dir>/report_card_verdict.json``; return its path.

    This is the canonical PER-STUDY location ``v2ecoli/workflow/analyses/
    comparison_matrix.py``'s ``_load_study_verdict``/``_config_verdicts_
    from_disk`` read (its ``_VERDICT_FILENAMES`` primary entry) -- unlike
    ``write_condition_verdict`` above (which nests under a per-condition
    subdirectory of a shared ``card_root``, the Gen-1 ``scripts/_compare``
    convention), this writes directly under the study's own directory, one
    file per study, matching the composite-substrate's one-study-per-config
    model (see ``comparison_cards.py``'s ``study_dir`` parameter).
    """
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    path = study_dir / "report_card_verdict.json"
    text = json.dumps(verdict, indent=2, ensure_ascii=False)
    if atomic_write_text is not None:
        atomic_write_text(path, text)
    else:
        path.write_text(text, encoding="utf-8")
    return path
