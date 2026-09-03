"""Test: with the gate corrected to >=, does the lineage stall, and what limits it?

Sibling of ``replisome_arrest_card`` on the same three-way split -- the
measurement lives in ``v2ecoli/library/gate_sufficiency.py``, this module is only
the Step.

Gated on the study declaring::

    report_card_refs:
      gate_sufficiency:
        out_root:    out/replisome-gate-sufficiency
        bundle_dir:  workspace/studies/replisome-gate-sufficiency/evidence
        expected_generations: 12

Every axis here is an ``acceptance_criterion`` pre-registered before the sweep
ran (see the study's ``preregistered:`` block, registered_at
2026-09-02T19:12:58Z, with ``criteria_match`` verified against these very
``pass_if`` values). There are no regression pins: nothing in this card was
chosen after seeing a number.

The discriminating axes are ``subunit_shortfall`` and ``ablation_relief``.
Study 1 could measure neither: under ``==`` every pool failed the comparison
whether short or in surplus, so "which pool limits" was unaskable. Together they
separate three explanations the previous study could not -- a mass-gated stall,
a stall limited by some other pool, and DnaG as the genuine binding constraint.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import gate_sufficiency as gs
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class GateSufficiencyCard(ReportCardStep):
    name = "gate_sufficiency"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        if not isinstance(cfg, dict):
            return None
        return cfg if cfg.get("out_root") and cfg.get("bundle_dir") else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None
        root = study.ws_root

        def _p(key):
            v = cfg[key]
            p = Path(v) if str(v).startswith("/") else (root / v)
            if not p.exists():
                raise FileNotFoundError(
                    f"report_card_refs.{self.name}.{key} does not exist: {p}")
            return p

        m = gs.measure(_p("out_root"), _p("bundle_dir"))
        exp = int(cfg.get("expected_generations", m["expected_generations"]))
        mech, perm, abl = (m["arms"]["mechanistic"], m["arms"]["permissive"],
                           m["arms"]["dnag-ablation"])
        b = TestBuilder(model_ref=study.study_name)

        b.add("Stall", check(
            "lineage_stalls", "Every mechanistic seed stops short of the run length",
            mech["divided_max"], value(exp, op="<"),
            severity="hard", units="generations",
            knob=["ecoli-chromosome-replication.mechanistic_replisome"],
            detail={"gate_class": "acceptance_criterion",
                    "n_stalled": f"{mech['n_stalled']}/{mech['n_seeds']}",
                    "stall_generations": mech["stall_generations"],
                    "discrimination":
                        "Graded on the MAXIMUM divided count across seeds, so one "
                        "surviving seed fails it. A model with adequate subunit "
                        "supply passes the run length and fails this."}))

        b.add("Stall", check(
            "stall_seed_consistency", "Stall generation is consistent across seeds",
            mech["stall_spread"], value(3, op="<="),
            severity="hard", units="generations",
            detail={"gate_class": "acceptance_criterion",
                    "stall_generations": mech["stall_generations"],
                    "note": "max - min across 8 seeds. Robustness at n=8, NOT "
                            "powered inference; no rank test was planned."}))

        b.add("Stall", check(
            "permissive_completes", "Permissive control runs the full length",
            perm["divided_min"], value(exp, op=">="),
            severity="hard", units="generations",
            detail={"gate_class": "acceptance_criterion",
                    "role": "control -- a failure here means the lineage dies for "
                            "a reason unrelated to the gate, refuting the causal "
                            "chain at its first link",
                    "n_seeds": perm["n_seeds"]}))

        b.add("Mechanism", check(
            "subunit_shortfall", "Some replisome pool is genuinely short at the stall",
            m["worst_margin"], value(0, op="<"),
            severity="hard", units="copies",
            detail={"gate_class": "acceptance_criterion",
                    "worst_margin": m["worst_margin"],
                    "per_seed_worst": {str(k): f"{v['pool']} {v['margin']:+d}"
                                       for k, v in m["per_seed_worst"].items()},
                    "interpretation":
                        "A NEGATIVE margin is a real shortfall. This is the axis "
                        "`==` made unmeasurable. All-positive margins would mean "
                        "the stall is gated by critical mass, not subunit supply."}))

        b.add("Mechanism", check(
            "dnag_is_limiting", "DnaG holds the worst margin in every stalled seed",
            m["limiting_pool_unanimous"], value(True, op="=="),
            severity="hard",
            detail={"gate_class": "acceptance_criterion",
                    "limiting_pool": m["limiting_pool"],
                    "pools_seen": m["limiting_pools_seen"],
                    "discrimination":
                        "Five other pools could have held the worst margin, and "
                        "HolA was close to marginal in the prior. Unanimity across "
                        "8 seeds, not a single-seed identification."}))

        b.add("Mechanism", check(
            "ablation_relief", "Dropping DnaG from the gate extends survival",
            m["ablation_relief_min"], value(0, op=">"),
            severity="hard", units="generations",
            knob=["ecoli-chromosome-replication.replisome_monomers_subunits"],
            detail={"gate_class": "acceptance_criterion",
                    "relief_per_seed": {str(k): v for k, v in m["ablation_relief"].items()},
                    "discrimination":
                        "The causal axis. Correlation (DnaG is scarcest) cannot "
                        "pass it; only removing DnaG from the requirement and "
                        "observing the stall lift can. Paired per seed."}))

        return b.build(), _render(m, exp)


def _render(m: dict, exp: int) -> str:
    mech = m["arms"]["mechanistic"]
    rows = "".join(
        f"<tr><td>{arm}</td><td align=right>{v['n_seeds']}</td>"
        f"<td align=right>{v['divided_min']}-{v['divided_max']}</td>"
        f"<td align=right>{v['n_stalled']}/{v['n_seeds']}</td>"
        f"<td align=right>{v['stall_generations'] or '-'}</td></tr>"
        for arm, v in m["arms"].items())
    worst = "".join(
        f"<tr><td>seed {k}</td><td>{v['pool']}</td>"
        f"<td align=right>{v['min_count']}</td>"
        f"<td align=right style='color:#8c2f22'>{v['margin']:+d}</td></tr>"
        for k, v in sorted(m["per_seed_worst"].items()))
    relief = ", ".join(f"seed {k}: {v:+d}" for k, v in sorted(m["ablation_relief"].items()))
    return f"""
<div style="font:14px system-ui">
  <h3>Replisome gate sufficiency &mdash; stall, and what limits it</h3>
  <p>With the gate corrected to <code>&gt;=</code>, <b>{mech['n_stalled']} of
     {mech['n_seeds']}</b> mechanistic seeds still stall, at generation
     {min(mech['stall_generations'])}&ndash;{max(mech['stall_generations'])}
     (spread <b>{mech['stall_spread']}</b>) of {exp}.</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>arm</th><th>seeds</th><th>divided</th><th>stalled</th><th>stall gens</th></tr>
    {rows}
  </table>
  <p style="margin-top:12px">Worst pool at each stall &mdash; <b>{m['limiting_pool']}</b>
     in {'every' if m['limiting_pool_unanimous'] else 'some'} seed:</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>seed</th><th align=left>pool</th><th>min count</th><th>margin</th></tr>
    {worst}
  </table>
  <p style="margin-top:12px">Removing DnaG from the gate's requirement list
     extends survival by {relief} generations.</p>
</div>"""
