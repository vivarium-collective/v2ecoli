"""Test: does replisome-subunit gating arrest a lineage, and is a pool actually short?

Sibling of ``genotype_build_integrity_card`` on the same three-way split — the
measurement lives in ``v2ecoli/library/replisome_arrest.py``, this module is
only the Step.

Gated on the study declaring::

    report_card_refs:
      replisome_arrest:
        mechanistic_dir: out/mechanistic-replisome-arrest/mechanistic
        permissive_dir:  out/mechanistic-replisome-arrest/permissive
        cache_dir:       out/cache
        expected_generations: 12

Why these axes. The study's original behaviour tests graded "did the lineage
reach 12 generations", which a null model passes vacuously — ANY broken model
fails to reach 12. The audit flagged it as ``one_sided_loose_primary`` and it
is kept, marked weak, because it was genuinely pre-registered. The axes that
discriminate are the paired comparison (same seed, same cache, only the gate
differs) and the subunit margin, which separates "the gate blocked initiation"
from "a subunit pool was actually exhausted" — the distinction the whole
investigation turns on.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, band, value

from v2ecoli.library import replisome_arrest as ra
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class ReplisomeArrestCard(ReportCardStep):
    name = "replisome_arrest"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        if not isinstance(cfg, dict):
            return None
        return cfg if cfg.get("mechanistic_dir") and cfg.get("permissive_dir") else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None

        # StudyContext.load takes the REPO root (it appends workspace/studies/),
        # and report_card_refs paths are repo-relative, so resolve against
        # ws_root itself. Using .parent silently resolved one level too high;
        # TestStep.invoke swallows the resulting error, so the card just went
        # missing rather than failing loudly.
        root = study.ws_root

        def _p(key):
            v = cfg[key]
            p = Path(v) if str(v).startswith("/") else (root / v)
            if not p.exists():
                raise FileNotFoundError(
                    f"report_card_refs.{self.name}.{key} does not exist: {p}")
            return p

        m = ra.measure(_p("mechanistic_dir"), _p("permissive_dir"), _p("cache_dir"))
        expected_gens = int(cfg.get("expected_generations", 12))

        b = TestBuilder(model_ref=study.study_name)

        # --- Pre-registered, and weak. Kept because it was genuinely stated
        # before the run; marked so no reader mistakes it for discrimination.
        b.add("Arrest", check(
            "arrest_preregistered", "Mechanistic lineage stops short of the run length",
            m["mechanistic_generations"], value(expected_gens, op="<"),
            severity="hard", units="generations",
            knob=["ecoli-chromosome-replication.mechanistic_replisome"],
            detail={
                "gate_class": "acceptance_criterion",
                "discrimination": "WEAK — a null model passes vacuously; any "
                                  "failure to reach the run length satisfies it. "
                                  "Retained because it was pre-registered, not "
                                  "because it discriminates.",
            }))

        # --- The discriminating axis: same seed, same cache, only the gate differs.
        b.add("Arrest", check(
            "paired_gap", "Permissive outlasts mechanistic (paired, same seed+cache)",
            m["generation_gap"], value(1, op=">="),
            severity="hard", units="generations",
            knob=["ecoli-chromosome-replication.mechanistic_replisome"],
            detail={
                "gate_class": "acceptance_criterion",
                "mechanistic_generations": m["mechanistic_generations"],
                "permissive_generations": m["permissive_generations"],
                "discrimination": "A model without subunit gating cannot produce "
                                  "a positive gap; both arms share one cache and "
                                  "a byte-identical initial state.",
            }))

        b.add("Arrest", check(
            "permissive_completes", "Permissive control runs the full length",
            m["permissive_generations"], value(expected_gens, op=">="),
            severity="hard", units="generations",
            detail={"gate_class": "acceptance_criterion",
                    "role": "control — a failure here refutes the causal chain "
                            "at its first link"}))

        # --- Post-hoc: pins the observed arrest point. NOT acceptance evidence.
        b.add("Arrest", check(
            "arrest_generation_pin", "Arrest generation (regression pin)",
            m["mechanistic_generations"], band(1, 3),
            severity="soft", units="generations",
            detail={
                "gate_class": "regression_pin",
                "note": "Band chosen AFTER observing arrest at generation 2. "
                        "Locks the behaviour against silent drift; never counts "
                        "as acceptance evidence.",
            }))

        # --- The mechanism axis: was a pool actually short?
        worst = m["worst_subunit_margin"]
        b.add("Mechanism", check(
            "subunit_shortfall", "Some replisome pool is short at the arrest",
            worst, value(0, op="<"),
            severity="hard", units="copies",
            knob=["ecoli-chromosome-replication.mechanistic_replisome"],
            detail={
                "gate_class": "acceptance_criterion",
                "limiting_pool": m["limiting_pool"],
                "n_pools_graded": m["n_pools_graded"],
                "margins": {v["label"]: v["margin"]
                            for v in m["subunit_margins"].values()},
                "interpretation":
                    "A NEGATIVE worst margin means a pool genuinely ran out. A "
                    "POSITIVE one means every pool was in surplus and the gate "
                    "blocked initiation for another reason — which refutes the "
                    "subunit-depletion hypothesis rather than supporting it.",
            }))

        html = _render(m, expected_gens)
        return b.build(), html


def _render(m: dict, expected_gens: int) -> str:
    rows = "".join(
        f"<tr><td>{v['label']}</td><td style='text-align:right'>{v['requirement_per_oric']}&times; oriC</td>"
        f"<td style='text-align:right'>{v['min_count']}</td>"
        f"<td style='text-align:right;color:{'#8c2f22' if v['margin'] < 0 else '#2c5f4a'}'>"
        f"{v['margin']:+d}</td></tr>"
        for v in m["subunit_margins"].values())
    short = m["worst_subunit_margin"]
    verdict = ("a pool was genuinely short" if short is not None and short < 0
               else "every pool was in surplus")
    return f"""
<div style="font:14px system-ui">
  <h3>Replisome gating — arrest and its cause</h3>
  <p>Mechanistic arm completed <b>{m['mechanistic_generations']}</b> generations;
     permissive completed <b>{m['permissive_generations']}</b> of {expected_gens}
     (gap <b>{m['generation_gap']:+d}</b>). Both arms share one cache and a
     byte-identical initial state.</p>
  <p>At the arresting generation ({m['arrest_generation']}),
     <b>{verdict}</b> — worst margin {short:+d} copies
     ({m['limiting_pool']}).</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align="left">pool</th><th>requirement</th><th>min count</th><th>margin</th></tr>
    {rows}
  </table>
</div>"""
