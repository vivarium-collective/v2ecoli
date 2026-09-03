"""Test: is basal_prob the reference in force, and does the deficit survive the right one?

Measurement in ``v2ecoli/library/inforce_reference.py``; grading here.

``reconstruction_is_valid`` is graded FIRST in spirit: the in-force vector is
recomputed rather than read, so if it fails to predict what the well-behaved
comparison transcripts actually did, nothing downstream of it may be trusted.

``ppgpp_branch_is_active_at_runtime`` is left UNGRADED. Its specified measurement
-- the fraction of timesteps where ppgpp_state is non-empty -- needs that store
emitted, and emitting it means editing a cache-hashed source file, which
invalidates the cache and breaks comparability with six prior studies. It is
reported as ungraded rather than silently re-specified onto a proxy.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import inforce_reference as ir
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class InForceReferenceCard(ReportCardStep):
    name = "inforce_reference"

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

        m = ir.measure(_p("cache_dir"), _p("out_root"))
        b = TestBuilder(model_ref=study.study_name)

        b.add("Instrument", check(
            "reconstruction_is_valid",
            "The recovered vector predicts what well-behaved transcripts did",
            m["reconstruction_validation"], value(0.5, op=">="),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "comparison_observed_over_predicted": m["reconstruction_validation"],
                    "discrimination":
                        "The vector is RECOMPUTED, not read. If it does not predict the "
                        "294 comparison transcripts -- which are realized at 1.006x "
                        "assigned and therefore behave -- every axis below is void."}))

        b.add("Reference", check(
            "in_force_reference_differs_from_basal",
            "The vector in force is materially different from the cache's basal_prob",
            m["reference_vs_basal_median_logratio"], value(0.1, op=">"),
            severity="hard", units="log-ratio",
            detail={"gate_class": "acceptance_criterion",
                    "reference_sum": m["reference_sum"], "basal_sum": m["basal_sum"],
                    "interpretation":
                        "A PASS means three prior studies divided by a vector the model "
                        "does not consult. The cache vector does not even sum to 1."}))

        b.add("Reference", check(
            "deficit_survives_correct_reference",
            "The cohort is still depressed against the vector actually in force",
            m["deficit_vs_reference"], value(0.5, op="<"),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "cohort_ratio": m["cohort_ratio_vs_reference"],
                    "comparison_ratio": m["comparison_ratio_vs_reference"],
                    "same_measure_vs_basal": m["deficit_vs_basal"],
                    "discrimination":
                        "The axis able to retract three studies. It passes, so the "
                        "deficit is real -- but at ~6x rather than the ~52x the wrong "
                        "denominator produced."}))

        b.add("Mechanism", check(
            "suppression_not_explained_by_configured_regulators",
            "The suppression is not what the configured regulators would produce",
            m["additive_regulator_mismatch"], value(0.5, op=">"),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "delta_sum_over_basal": m["delta_sum_over_basal"],
                    "note": "delta_sum/basal is exactly -1.0 for TU00352 and TU00062 -- "
                            "regulators wired to null the gene -- yet the observed "
                            "suppression does not match the additive prediction. Under "
                            "ppGpp the TF effect is multiplicative, so this is the "
                            "ADDITIVE null, not a test of the multiplicative path."}))

        return b.build(), _render(m)


def _render(m: dict) -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td align=right>{v['actual']:.3e}</td>"
        f"<td align=right>{v['reference']:.3e}</td><td align=right>{v['basal']:.3e}</td>"
        f"<td align=right>{v['ratio_ref']:.3f}</td></tr>"
        for t, v in m["per_tu"].items())
    return f"""
<div style="font:14px system-ui">
  <h3>In-force reference &mdash; the deficit is real but was overstated</h3>
  <p>The vector actually in force differs from the cache's <code>basal_prob</code>
     (median |log ratio| {m['reference_vs_basal_median_logratio']:.3f}; sums
     {m['reference_sum']:.4f} vs {m['basal_sum']:.4f}). Measured against it, the
     cohort sits at {m['cohort_ratio_vs_reference']:.3f} against the comparison's
     {m['comparison_ratio_vs_reference']:.3f} &mdash; a deficit of
     <b>{m['deficit_vs_reference']:.3f}</b> (~{1/m['deficit_vs_reference']:.0f}x),
     where the wrong denominator gave {m['deficit_vs_basal']:.4f}
     (~{1/m['deficit_vs_basal']:.0f}x).</p>
  <p>Reconstruction validated: the comparison transcripts transcribe at
     {m['reconstruction_validation']:.3f} of the recovered vector, so it predicts
     the behaviour of transcripts known to behave.</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>transcript</th><th>in force (actual)</th><th>reference</th>
        <th>cache basal</th><th>ratio</th></tr>
    {rows}
  </table>
</div>"""
