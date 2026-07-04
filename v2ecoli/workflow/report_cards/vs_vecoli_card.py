# v2ecoli/workflow/report_cards/vs_vecoli_card.py
from __future__ import annotations

import json
from pathlib import Path

from v2ecoli.library.report_card import render_verdict_html
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class VsVecoliCard(ReportCardStep):
    name = "vs_vecoli"

    def _verdict_path(self, study: StudyContext) -> "Path | None":
        rel = (study.spec.get("report_card_refs") or {}).get("vs_vecoli")
        if not rel:
            return None
        p = Path(rel) if str(rel).startswith("/") else (study.ws_root / rel)
        return p if p.is_file() else None

    def applies(self, study: StudyContext) -> bool:
        return self._verdict_path(study) is not None

    def build(self, study: StudyContext):
        vp = self._verdict_path(study)
        if vp is None:
            return None
        vjson = json.loads(vp.read_text(encoding="utf-8"))
        title = vjson.get("title") or (
            f"vEcoli ↔ v2ecoli — {study.spec.get('name', study.study_name)}")
        html = render_verdict_html(vjson, title=title)
        return vjson, html
