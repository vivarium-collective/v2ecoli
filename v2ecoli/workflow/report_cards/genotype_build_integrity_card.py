"""Gen-2 report card: does a ParCa-level knockout reconstruct into a sane sim_data?

Sibling of ``acetate_overflow_card`` and built on the same three-way split: the
science core lives in ``v2ecoli/library/genotype_build.py``, grading + rendering in
``v2ecoli/library/report_card.py``, and this module is only the Step.

Gated on the study declaring ``report_card_refs.genotype_build_integrity`` with the
gene ids to knock out, e.g.::

    report_card_refs:
      genotype_build_integrity:
        gene_ids: [EG10526]        # lacY
        mode: fast                 # optional; omit for the fit-free structural pass
        parca_state: out/...       # optional; a pre-built state for the fit axes

The card generates its own variant bundle and builds ``raw_data`` for both arms, so
the four structural axes need no ParCa run and cost seconds. The fit axes grade only
when a ``parca_state`` is supplied; otherwise they stay ungraded, which is the honest
outcome rather than a fabricated pass.
"""
from __future__ import annotations

from pathlib import Path

from v2ecoli.library import genotype_build as gb
from v2ecoli.library.report_card import grade_card, render_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class GenotypeBuildIntegrityCard(ReportCardStep):
    name = "genotype_build_integrity"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        if not isinstance(cfg, dict):
            return None
        return cfg if cfg.get("gene_ids") else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None

        state = cfg.get("parca_state")
        if state:
            p = Path(state) if str(state).startswith("/") else (study.ws_root / state)
            state = p if p.is_file() else None

        # Bundle generation writes into out/, never the study dir: the artifacts are
        # large and derived, and the study dir is tracked.
        workdir = study.ws_root.parent / "out" / "genotype_build" / study.study_name

        card, reference = gb.build(
            cfg["gene_ids"], workdir=workdir,
            parca_state=state, mode=cfg.get("mode"))
        report = grade_card(card, reference)
        vjson = verdict_json(
            report,
            model_ref=reference["stimulus"]["measured_model"],
            reference_model=reference["stimulus"]["reference_model"])
        vjson["title"] = reference["title"]
        vjson["genotype"] = card["genotype"]
        # No `generated=`: the runner commits this output, so it must stay
        # byte-deterministic across re-renders. The genotype id is content-addressed,
        # so an identical genotype re-renders identically.
        html = render_html(card, reference,
                           model_ref=reference["stimulus"]["measured_model"])
        return vjson, html
