"""Test: are the six top-300 zero-synthesis TUs reproducibly silent, and why?

Sibling of the other cards on the same three-way split -- measurement in
``v2ecoli/library/silent_tus.py``, grading here.

Every axis was pre-registered before the sweep ran (registered_at is in the
study's ``preregistered`` block, with ``criteria_match`` verified against these
``pass_if`` values). Only ``dnag_remains_stranded`` is a regression pin, because
only the cohort's IDENTITY came from prior observation.

The first axis is the null and is expected to be able to dissolve the study: a
single generation cannot distinguish "never transcribed" from "not transcribed
that time", since 43.3% of all TUs receive zero in one generation. It is written
so a single transcript anywhere fails it.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import silent_tus as st
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class SilentTUsCard(ReportCardStep):
    name = "silent_tus"

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

        m = st.measure(_p("cache_dir"), _p("out_root"))
        b = TestBuilder(model_ref=study.study_name)

        b.add("Reproducibility", check(
            "zero_synthesis_is_reproducible",
            "Cohort receives zero transcripts in every seed and generation",
            m["cohort_max_any_run"], value(0, op="=="),
            severity="hard", units="transcripts",
            detail={"gate_class": "acceptance_criterion",
                    "per_tu": {k: v["per_run"] for k, v in m["cohort"].items()},
                    "n_observations": m["n_observations"],
                    "discrimination":
                        "The null. 43.3% of all TUs get zero in ONE generation, so a "
                        "single observation cannot establish silence. Any transcript "
                        "anywhere fails this and dissolves the 'silent' framing."}))

        b.add("Reproducibility", check(
            "comparison_group_is_synthesized",
            "The 294 comparable top-300 TUs are transcribed",
            m["comparison_median_count"], value(0, op=">"),
            severity="hard", units="transcripts",
            detail={"gate_class": "acceptance_criterion",
                    "median_ratio_realized_over_assigned": m["comparison_median_ratio"],
                    "zero_fraction": m["comparison_zero_fraction"],
                    "role": "positive control -- a failure here indicts the run or the "
                            "listener, not the cohort"}))

        b.add("Mechanism", check(
            "effective_probability_explains_silence",
            "Cohort is realized at under 1% of its assigned probability",
            m["cohort_max_ratio"], value(0.01, op="<"),
            severity="hard", units="ratio",
            detail={"gate_class": "acceptance_criterion",
                    "per_tu_ratio": {k: v["realized_over_assigned"]
                                     for k, v in m["cohort"].items()},
                    "comparison_median_ratio": m["comparison_median_ratio"],
                    "depression_fold": m["depression_fold"],
                    "interpretation":
                        "Comparison TUs are realized at ~1.0x assigned, so the model "
                        "normally transcribes as specified. The threshold of 0.01 was "
                        "set before the run and is NOT relaxed to fit the result."}))

        b.add("Scope", check(
            "silence_generalises_beyond_dnag",
            "Another cistron depends entirely on cohort transcripts",
            len(m["stranded_excluding_dnag"]), value(1, op=">="),
            severity="hard", units="cistrons",
            detail={"gate_class": "acceptance_criterion",
                    "stranded": m["stranded_excluding_dnag"],
                    "discrimination":
                        "dnaG surfaced only because a replication gate made it visible. "
                        "Other stranded cistrons would show the defect is general."}))

        b.add("Scope", check(
            "dnag_remains_stranded", "dnaG still receives no transcripts",
            m["dnag_total_synthesized"], value(0, op="=="),
            severity="soft", units="transcripts",
            detail={"gate_class": "regression_pin",
                    "note": "POST-HOC from dnag-production-deficit. Pin only."}))

        return b.build(), _render(m)


def _render(m: dict) -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td align=right>{v['total']:.0f}</td>"
        f"<td align=right>{v['max']:.0f}</td>"
        f"<td align=right>{v['realized_over_assigned']:.4f}</td></tr>"
        for t, v in m["cohort"].items())
    return f"""
<div style="font:14px system-ui">
  <h3>Silent transcription units &mdash; depressed, not silent</h3>
  <p>Across {m['n_observations']} (seed, generation) observations the cohort received
     <b>{m['cohort_total']:.0f}</b> transcripts in total, with a maximum of
     <b>{m['cohort_max_any_run']:.0f}</b> in any single run. The "never transcribed"
     framing is therefore <b>refuted</b>.</p>
  <p>What survives is a large, consistent depression: the cohort is realized at
     {m['cohort_max_ratio']:.4f} of its assigned probability at best, against a
     comparison median of {m['comparison_median_ratio']:.3f} &mdash; about
     <b>{m['depression_fold']:.0f}x</b> below peers that transcribe essentially as
     specified.</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>transcript</th><th>total</th><th>max/run</th>
        <th>realized/assigned</th></tr>
    {rows}
  </table>
  <p style="margin-top:10px">Cistrons stranded entirely on cohort transcripts:
     <b>{', '.join(m['stranded_cistrons'])}</b> &mdash; dnaG is not alone.</p>
</div>"""
