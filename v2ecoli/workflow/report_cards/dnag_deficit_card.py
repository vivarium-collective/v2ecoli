"""Test: where is DnaG lost, and is the model reproducing its own fit?

Sibling of ``replisome_arrest_card`` / ``gate_sufficiency_card`` on the same
three-way split -- measurement in ``v2ecoli/library/dnag_deficit.py``, grading
here, rendering in ``_render``.

Axis classification is deliberately mixed and must stay that way. Three axes
were genuinely open when the study was designed and are graded as
``acceptance_criterion``; three were already measured during scoping and are
``regression_pin``. A pin locks observed behaviour against drift but cannot
confirm anything, and presenting one as acceptance evidence is the exact
post-hoc failure the audit hunts.

``transcription_not_lossy`` is stated so the investigation's OWN standing
hypothesis -- that the rpsU-promoter dedup starves dnaG of transcription -- is
falsifiable: that hypothesis predicts the axis FAILS. Writing it the other way
round would have made a refutation unreportable.
"""
from __future__ import annotations

from pathlib import Path

from viva_superpowers import TestBuilder, check, value

from v2ecoli.library import dnag_deficit as dd
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class DnaGDeficitCard(ReportCardStep):
    name = "dnag_deficit"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        return cfg if isinstance(cfg, dict) and cfg.get("cache_dir") else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None
        root = study.ws_root

        def _p(key, required=True):
            v = cfg.get(key)
            if v is None:
                return None
            p = Path(v) if str(v).startswith("/") else (root / v)
            if required and not any(Path(p).parent.glob(Path(p).name)) and not p.exists():
                raise FileNotFoundError(f"report_card_refs.{self.name}.{key}: {p}")
            return p

        m = dd.measure(_p("cache_dir"), _p("bundle_glob"), _p("proteome_script"),
                       fixture=_p("fixture"))
        # Realized transcription: what the simulation ACTUALLY made. basal_prob is
        # overridden for idx_rprotein TUs and cannot answer this.
        rt = dd.realized_transcription(_p("cache_dir"), cfg["transcription_run_glob"])
        b = TestBuilder(model_ref=study.study_name)

        b.add("Execution", check(
            "simulation_reproduces_parca_target",
            "Simulation reproduces ParCa's fitted DnaG count",
            m["sim_vs_parca"], value(0.5, op=">="),
            severity="hard", units="ratio",
            knob=["ecoli-polypeptide-initiation.translation_efficiencies"],
            detail={"gate_class": "acceptance_criterion",
                    "parca_fitted": (m["parca_expected"] or {}).get("count"),
                    "simulated_mean": m["simulated"].get("mean"),
                    "discrimination":
                        "Compares the run against the model's OWN target, not "
                        "against experiment, so a calibration gap cannot pass it "
                        "and an execution gap cannot hide behind one."}))

        b.add("Localisation", check(
            "dnag_transcript_is_synthesized",
            "dnaG's transcript is actually synthesised during a run",
            rt.get("dnag_total_synthesized"), value(0, op=">"),
            severity="hard", units="transcripts/generation",
            detail={"gate_class": "acceptance_criterion",
                    "per_transcript": (rt.get("genes") or {}).get("dnaG"),
                    "tu00352_mrna": rt.get("tu00352_mrna"),
                    "discrimination":
                        "Measured from count_rna_synthesized DURING a run. Replaces an "
                        "axis that ranked basal_prob -- a parameter that is OVERRIDDEN "
                        "for idx_rprotein TUs and reached the opposite conclusion."}))

        b.add("Localisation", check(
            "operon_partners_share_transcription",
            "Cistrons on TU00352 receive comparable transcription",
            rt.get("operon_transcript_spread"), value(10.0, op="<"),
            severity="hard", units="fold",
            detail={"gate_class": "acceptance_criterion",
                    "transcripts_made": {g: v["total_synthesized"]
                                         for g, v in (rt.get("genes") or {}).items()},
                    "per_transcript": {g: v["per_transcript"]
                                       for g, v in (rt.get("genes") or {}).items()},
                    "discrimination":
                        "Graded on transcripts actually made, so it needs no shared-mRNA-"
                        "pool assumption -- the false premise that made a transcriptional "
                        "divergence look translational. A FAIL means the partners are "
                        "supplied via alternative transcripts while dnaG is not."}))

        lit = m["literature"]
        pin = (m["parca_expected"] or {}).get("count")
        b.add("Calibration", check(
            "dnag_below_literature", "ParCa's DnaG is below the 4-dataset median",
            m.get("parca_vs_lit"), value(1.0, op="<"),
            severity="soft", units="ratio",
            detail={"gate_class": "regression_pin",
                    "per_dataset": {k: lit.get(k) for k in
                                    ("Schmidt", "Soufi", "Mori", "Li")},
                    "median": lit.get("median"), "parca": pin,
                    "note": "POST-HOC (measured while scoping). Schmidt alone reads "
                            "DnaG far below the other three, so a single-dataset "
                            "check points the wrong way."}))

        sim = m["simulated"]
        b.add("Maintenance", check(
            "dnag_not_maintained", "DnaG spends much of the lineage at zero",
            sim.get("frac_zero"), value(0.25, op=">"),
            severity="soft", units="fraction",
            detail={"gate_class": "regression_pin",
                    "per_generation": {str(g): round(v["frac_zero"], 3)
                                       for g, v in (sim.get("per_generation") or {}).items()},
                    "note": "POST-HOC. Per-generation so a decline is visible."}))

        b.add("Maintenance", check(
            "supply_below_gate_demand",
            "Supply sits below the gate's DnaG demand much of the time",
            m.get("frac_below_gate_demand"), value(0.25, op=">"),
            severity="soft", units="fraction",
            detail={"gate_class": "regression_pin",
                    "gate_demand_per_oric": m.get("gate_demand_per_oric"),
                    "note": "POST-HOC relative to study 2, which already showed the "
                            "stall. Mechanical reason the option cannot be enabled."}))

        return b.build(), _render(m, rt)


def _render(m: dict, rt: dict | None = None) -> str:
    lit = m["literature"]
    chain = m["chain"]
    rows = "".join(
        f"<tr><td>{c['step']}</td><td>{c['quantity']}</td>"
        f"<td align=right>{c['value']:.3e}</td>"
        f"<td align=right>{c['median_all']:.3e}</td>"
        f"<td align=right style='color:{'#8c2f22' if c['percentile'] < 10 else '#2c5f4a'}'>"
        f"{c['percentile']:.1f}%</td></tr>"
        for k, c in chain.items() if k != "operon")
    ds = "".join(f"<tr><td>{k}</td><td align=right>{lit.get(k):,.0f}</td></tr>"
                 for k in ("Schmidt", "Soufi", "Mori", "Li") if lit.get(k) is not None)
    op = "".join(f"<tr><td>{g}</td><td align=right>{r:.2f}x</td></tr>"
                 for g, r in (m.get("operon_ratios") or {}).items() if r)
    sim = m["simulated"]
    parca = (m["parca_expected"] or {}).get("count")
    return f"""
<div style="font:14px system-ui">
  <h3>DnaG production deficit &mdash; where the protein is lost</h3>
  <p>Literature median <b>{lit.get('median'):,.0f}</b> copies; ParCa fits
     <b>{parca:,.0f}</b> ({m.get('parca_vs_lit'):.2f}x); the simulation delivers
     mean <b>{sim.get('mean'):.2f}</b> &mdash;
     <b>{1/m['sim_vs_parca']:.0f}x below the model's own target</b>, at zero for
     {sim.get('frac_zero'):.0%} of the lineage.</p>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th align=left>step</th><th align=left>quantity</th><th>dnaG</th>
        <th>median</th><th>percentile</th></tr>
    {rows}
  </table>
  <p style="margin-top:12px"><b>Transcription IS the lossy step.</b> Measured during a
     run, dnaG receives {(rt or {}).get('dnag_total_synthesized', '?')} transcripts per
     generation; TU00352 mRNA is absent at
     {(((rt or {}).get('tu00352_mrna') or {}).get('frac_zero', 0))*100:.0f}% of timesteps.
     Its operon partners escape via alternative transcripts:
     {', '.join(f"{g} {v['total_synthesized']:.0f}" for g, v in ((rt or {}).get('genes') or {}).items())}.
     The translation-efficiency percentile below is DOWNSTREAM of this block &mdash; an
     mRNA that is never made cannot be translated.</p>
  <div style="display:flex;gap:28px;margin-top:10px">
    <div><b>Literature (copies)</b>
      <table style="border-collapse:collapse;font-size:13px">{ds}</table></div>
    <div><b>Operon partners (model/lit)</b>
      <table style="border-collapse:collapse;font-size:13px">{op}</table>
      <div style="font-size:12px;color:#6b6b66;max-width:280px">One transcript,
        one mRNA pool &mdash; so this spread localises the loss downstream.</div></div>
  </div>
</div>"""
