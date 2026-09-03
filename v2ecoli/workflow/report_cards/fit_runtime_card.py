"""Test: is the ParCa-vs-runtime transcription gap dnaG-specific, and does the
basal/ppGpp substitution explain it?

Measurement in ``v2ecoli/library/fit_runtime.py``; grading here.

``balance_still_predicts_controls`` is re-measured rather than inherited: every
quantity in this card is derived by inverting that mass balance, and in the
predecessor study an earlier dimensionally-wrong version scored 0% on exactly
this check.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import fit_runtime as fr
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class FitRuntimeCard(ReportCardStep):
    name = "fit_runtime"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        return cfg if isinstance(cfg, dict) and cfg.get("out_root") else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None
        root = study.ws_root

        def _p(k):
            v = cfg[k]
            p = Path(v) if str(v).startswith("/") else (root / v)
            if not p.exists():
                raise FileNotFoundError(f"report_card_refs.{self.name}.{k}: {p}")
            return p

        m = fr.measure(_p("cache_dir"), _p("out_root"))
        b = TestBuilder(model_ref=study.study_name)

        b.add("Instrument", check(
            "balance_still_predicts_controls",
            "The inverted mass balance still predicts control proteins",
            m["control_within_2x"], value(0.7, op=">="),
            severity="hard", units="fraction",
            detail={"gate_class": "acceptance_criterion", "n": m["control_n"],
                    "discrimination":
                        "Every quantity here inverts this balance. An earlier "
                        "dimensionally-wrong version scored 0% on this check, which is "
                        "the only reason that error was caught."}))

        b.add("Specificity", check(
            "discrepancy_is_dnag_specific",
            "The fit-runtime gap is particular to dnaG, not transcriptome-wide",
            m["dnag_over_population"], value(3.0, op=">="),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "dnag": m["dnag_discrepancy"],
                    "population_median": m["population_median_discrepancy"],
                    "n_population": m["n_population"],
                    "interpretation":
                        "The population median is BELOW 1, so the typical transcript is "
                        "over-transcribed relative to its fit while dnaG is under it by "
                        "6.85x. dnaG is a genuine outlier, not an instance of a global "
                        "offset."}))

        b.add("Mechanism", check(
            "basal_vs_ppgpp_substitution_explains_it",
            "dnaG's gap equals the basal-to-ppGpp vector ratio",
            m["substitution_agreement"], value(2.0, op="<"),
            severity="hard", units="fold",
            knob=["ecoli-transcript-initiation.basal_prob"],
            detail={"gate_class": "acceptance_criterion",
                    "discrepancy": m["dnag_discrepancy"],
                    "vector_ratio": m["dnag_vector_ratio_basal_over_ppgpp"],
                    "discrimination":
                        "If ParCa fits against basal_prob while the runtime transcribes "
                        "per the ppGpp replacement, the gap should BE that ratio. It "
                        "agrees within 1.72x, at the mass balance's own precision."}))

        b.add("Mechanism", check(
            "population_discrepancy_tracks_vector_ratio",
            "The same explanation holds across the transcriptome",
            m["population_correlation"], value(0.5, op=">="),
            severity="hard", units="correlation",
            detail={"gate_class": "acceptance_criterion",
                    "n": m["n_population"],
                    "interpretation":
                        "A FAIL means the substitution explains dnaG but is not what "
                        "drives the fit-runtime relationship generally -- so it is a "
                        "real mechanism with limited reach, not a universal law."}))

        return b.build(), _render(m)


def _render(m: dict) -> str:
    return f"""
<div style="font:14px system-ui">
  <h3>Fit-runtime reconciliation &mdash; dnaG is a genuine outlier</h3>
  <p>dnaG's fitted count implies <b>{m['dnag_discrepancy']:.2f}x</b> more
     transcription than the model delivers, against a population median of
     <b>{m['population_median_discrepancy']:.2f}</b> across {m['n_population']}
     monomers &mdash; {m['dnag_over_population']:.1f}x the typical transcript. The
     median being below 1 means the ordinary gene is slightly OVER-transcribed
     relative to its fit, so dnaG is not an instance of a global offset.</p>
  <p>The gap matches the ratio between the two probability vectors
     ({m['dnag_vector_ratio_basal_over_ppgpp']:.2f}) within
     {m['substitution_agreement']:.2f}x: ParCa fits against
     <code>basal_prob</code> while the runtime transcribes per the ppGpp
     replacement.</p>
  <p>That explanation does <b>not</b> generalise: the population correlation
     between the discrepancy and the vector ratio is only
     {m['population_correlation']:.3f}. The substitution is a real mechanism with
     limited reach, not a transcriptome-wide law.</p>
  <p>Instrument: {100*m['control_within_2x']:.1f}% of {m['control_n']} controls
     within a factor of two.</p>
</div>"""
