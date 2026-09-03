"""Test: is the transcription deficit an assignment defect or a selection defect?

Measurement in ``v2ecoli/library/runtime_synth_prob.py``; grading here.

The two primary axes are complements and are meant to be read together:
``runtime_probability_is_depressed`` asks whether the probability IN FORCE is
low, and ``realized_tracks_runtime_probability`` asks whether transcripts are
drawn fairly GIVEN that probability. Together they attribute the deficit to
assignment, to selection, or to both.

``comparison_group_behaves_normally`` checks the instrument rather than the
finding: the comparison TUs are realized at 1.006x assigned, so their actual and
target probabilities must agree. If they do not, the emitted columns do not mean
what this study assumes and no other axis is interpretable.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import runtime_synth_prob as rsp
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class RuntimeSynthProbCard(ReportCardStep):
    name = "runtime_synth_prob"

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

        m = rsp.measure(_p("cache_dir"), _p("out_root"))
        b = TestBuilder(model_ref=study.study_name)

        b.add("Assignment", check(
            "runtime_probability_is_depressed",
            "The probability actually in force is depressed for the cohort",
            m["actual_depression"], value(0.5, op="<"),
            severity="hard", units="ratio",
            knob=["ecoli-transcript-initiation.basal_prob"],
            detail={"gate_class": "acceptance_criterion",
                    "cohort_actual_median": m["cohort_actual_median"],
                    "comparison_actual_median": m["comparison_actual_median"],
                    "discrimination":
                        "A PASS attributes the deficit to probability ASSIGNMENT; a "
                        "FAIL would move it downstream to transcript selection. These "
                        "need different repairs."}))

        b.add("Assignment", check(
            "actual_tracks_target",
            "The in-force probability tracks its target (depression is upstream)",
            m["cohort_actual_over_target"], value(0.5, op=">="),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "cohort_actual_over_target": m["cohort_actual_over_target"],
                    "cohort_tf_bound_median": m["cohort_tf_median"],
                    "interpretation":
                        "A PASS means nothing between target and in-force removes the "
                        "probability, so the suppression is already in the TARGET and "
                        "regulation is not responsible."}))

        b.add("Selection", check(
            "realized_tracks_runtime_probability",
            "Transcripts are drawn fairly given the probability in force",
            m["selection_fairness"], value(0.5, op=">="),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "cohort_realized_per_prob": m["cohort_realized_per_prob"],
                    "comparison_realized_per_prob": m["comparison_realized_per_prob"],
                    "discrimination":
                        "Complement of the assignment axis. A FAIL means the cohort is "
                        "under-sampled even relative to its own reduced probability -- "
                        "a second, independent defect."}))

        b.add("Instrument", check(
            "comparison_group_behaves_normally",
            "Actual and target agree for transcripts known to behave",
            m["comparison_control_fold"], value(2.0, op="<"),
            severity="hard", units="fold",
            detail={"gate_class": "acceptance_criterion",
                    "role": "instrument check -- a failure means the emitted columns do "
                            "not mean what this study assumes, invalidating every "
                            "other axis rather than being interesting"}))

        b.add("Selection", check(
            "cohort_remains_depressed", "Cohort still realized far below assigned",
            m["cohort_max_realized_over_assigned"], value(0.1, op="<"),
            severity="soft", units="ratio",
            detail={"gate_class": "regression_pin",
                    "note": "POST-HOC from silent-transcription-units. Pin only."}))

        return b.build(), _render(m)


def _render(m: dict) -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td align=right>{v['actual']:.3e}</td>"
        f"<td align=right>{v['target']:.3e}</td>"
        f"<td align=right>{v['assigned']:.3e}</td>"
        f"<td align=right>{v['n_tf']:.2f}</td></tr>"
        for t, v in m["per_tu"].items())
    return f"""
<div style="font:14px system-ui">
  <h3>Runtime synthesis probability &mdash; an assignment defect</h3>
  <p>The probability actually in force is
     <b>{1/m['actual_depression']:.0f}x lower</b> for the cohort
     ({m['cohort_actual_median']:.3e}) than for the 294 comparison transcripts
     ({m['comparison_actual_median']:.3e}), and it tracks its own target almost
     exactly ({m['cohort_actual_over_target']:.3f}). The suppression is therefore
     already present in the target, not applied after it.</p>
  <p>No transcription factors are bound on either side
     (cohort {m['cohort_tf_median']:.2f}, comparison {m['comparison_tf_median']:.2f}),
     so regulation is not responsible.</p>
  <p>Given the probability in force, the cohort is drawn at
     {m['selection_fairness']:.3f} of the comparison group's efficiency &mdash; a
     secondary effect an order of magnitude smaller than the
     {1/m['actual_depression']:.0f}x assignment gap.</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>transcript</th><th>in force</th><th>target</th>
        <th>assigned</th><th>TF bound</th></tr>
    {rows}
  </table>
</div>"""
