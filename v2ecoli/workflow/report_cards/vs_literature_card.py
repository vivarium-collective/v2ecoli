# v2ecoli/workflow/report_cards/vs_literature_card.py
"""The basal ``vs_literature`` report card as a Gen 2 ``ReportCardStep``.

Grades the blessed v2ecoli baseline against curated experimental + theoretical
reference values from the ecoli-sources ``validation_data`` bundle (the PR #134
reference-agnostic grader pointed at literature instead of a pinned run). Unlike
``tests`` / ``vs_vecoli``, this card *builds* its verdict live via
``v2ecoli.library.vs_literature`` — the same functions the standalone
``scripts/render_basal_vs_literature.py`` and ``tests/test_basal_vs_literature.py``
use — so the dashboard-native card and the golden test grade identically.

Measured (model) side: the committed
``tests/fixtures/population_phenotype_basal/model_*.json`` fixtures (the
reproducibility boundary; not a live zarr). A study opts in by declaring
``report_card_refs.vs_literature: <fixtures dir>`` in its ``study.yaml``; the card
``applies`` only when that dir resolves and holds ``model_physiology.json``.
"""
from __future__ import annotations

from pathlib import Path

from v2ecoli.library import vs_literature as vsl
from v2ecoli.library.report_card import grade_card, render_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class VsLiteratureCard(ReportCardStep):
    name = "vs_literature"

    def _fixtures_dir(self, study: StudyContext) -> "Path | None":
        """The model fixtures dir declared by ``report_card_refs.vs_literature``
        (relative to the workspace root, or absolute), or None if absent /
        missing its ``model_physiology.json``."""
        rel = (study.spec.get("report_card_refs") or {}).get("vs_literature")
        if not rel:
            return None
        p = Path(rel) if str(rel).startswith("/") else (study.ws_root / rel)
        return p if (p / "model_physiology.json").is_file() else None

    def applies(self, study: StudyContext) -> bool:
        return self._fixtures_dir(study) is not None

    def build(self, study: StudyContext):
        fdir = self._fixtures_dir(study)
        if fdir is None:
            return None
        card, reference, model = vsl.build(model_json=fdir / "model_physiology.json")
        report = grade_card(card, reference)
        model_ref = f"v2ecoli baseline ({model['ensemble']})"
        vjson = verdict_json(report, model_ref=model_ref,
                             reference_model=reference["stimulus"]["reference_model"])
        vjson["title"] = reference["title"]
        # Reproduce the standalone card's rich HTML (plots + coverage + findings)
        # rather than the minimal render_verdict_html — but deterministic (no
        # generated timestamp) since the runner commits the output.
        html = render_html(card, reference, model_ref=model_ref)
        return vjson, html
