"""Gen-2 report card: aerobic acetate overflow (acetate-carbon yield vs growth rate).

Sibling of ``vs_literature_card`` and built on the same three-way split: the science
core lives in ``v2ecoli/library/overflow.py``, grading + rendering in
``v2ecoli/library/report_card.py``, and this module is only the Step.

Gated on ``report_card_refs.acetate_overflow`` pointing at a directory holding the
baked GUR-titration arm (``model_overflow_baseline.json``), so the card grades
run-free from a ``StudyContext``.
"""
from __future__ import annotations

from pathlib import Path

from v2ecoli.library import overflow as ovf
from v2ecoli.library.report_card import grade_card, render_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class AcetateOverflowCard(ReportCardStep):
    name = "acetate_overflow"

    def _fixtures_dir(self, study: StudyContext) -> "Path | None":
        rel = (study.spec.get("report_card_refs") or {}).get("acetate_overflow")
        if not rel:
            return None
        p = Path(rel) if str(rel).startswith("/") else (study.ws_root / rel)
        return p if (p / "model_overflow_baseline.json").is_file() else None

    def applies(self, study: StudyContext) -> bool:
        return self._fixtures_dir(study) is not None

    def build(self, study: StudyContext):
        fdir = self._fixtures_dir(study)
        if fdir is None:
            return None
        card, reference, model = ovf.build(model_json=fdir / "model_overflow_baseline.json")
        report = grade_card(card, reference)
        model_ref = reference["stimulus"]["measured_model"]
        vjson = verdict_json(report, model_ref=model_ref,
                             reference_model=reference["stimulus"]["reference_model"])
        vjson["title"] = reference["title"]
        # No `generated=`: the runner commits this output, so it must stay
        # byte-deterministic across re-renders.
        html = render_html(card, reference, model_ref=model_ref)
        return vjson, html
