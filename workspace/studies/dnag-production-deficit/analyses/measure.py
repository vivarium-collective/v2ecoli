"""Regenerate every derived field of the study from the ParCa cache, the
proteomics datasets and the reused multi-generation bundles.

Same pattern as the sync_study_state.py in the two preceding studies, and
idempotent for the same reason: a hand-set number goes stale the first time
anything upstream changes, and nothing here is hand-set.

This study runs no new simulation. Its simulated arm reuses the permissive
(healthy, unperturbed) lineage from replisome-gate-sufficiency -- same cache,
same composite, 3 seeds, 114k timesteps -- because re-running an identical
lineage would cost 3 hours to reproduce numbers already in hand. That reuse is
recorded as a limitation rather than presented as fresh replication.

Usage::

    python workspace/studies/dnag-production-deficit/analyses/measure.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

STUDY = STUDY_DIR.name
CARD = "dnag_deficit"
CANONICAL_RUN = f"{STUDY}__analysis"

AUTHORED_AXES = {"design_status": "complete", "implementation_status": "complete",
                 "expert_review_status": "pending"}


def _apply(measured, pass_if: dict) -> bool:
    op, want = str(pass_if.get("op", "")).strip(), pass_if.get("value")
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    if op not in ops:
        raise ValueError(f"unsupported op {op!r}")
    return bool(ops[op](measured, want))


def measure_tests(m: dict) -> dict:
    sim, lit = m["simulated"], m["literature"]
    parca = (m["parca_expected"] or {}).get("count")
    ratios = {g: r for g, r in (m.get("operon_ratios") or {}).items() if r}
    spread = (max(ratios.values()) / min(ratios.values())) if len(ratios) > 1 else None
    tx = m["chain"]["transcription"]
    return {
        "simulation-reproduces-parca-target": (
            m["sim_vs_parca"],
            f"Simulation mean {sim['mean']:.2f} vs ParCa's fitted {parca:,.0f} copies "
            f"= {m['sim_vs_parca']:.4f} ({1/m['sim_vs_parca']:.0f}x below its own target)."),
        "transcription-is-not-the-lossy-step": (
            tx["percentile"],
            f"TU00352 basal_prob {tx['value']:.3e} is at the {tx['percentile']:.1f}th "
            f"percentile of {tx['n']} transcripts (median {tx['median_all']:.3e}); "
            f"{tx['value']/tx['median_all']:.0f}x the median."),
        "operon-partners-diverge": (
            spread,
            "model/literature ratios on the shared transcript TU00352: "
            + ", ".join(f"{g} {r:.2f}x" for g, r in ratios.items())
            + f" -> {spread:.2f}x spread."),
        "dnag-below-literature": (
            m["parca_vs_lit"],
            f"ParCa {parca:,.0f} vs 4-dataset median {lit['median']:,.0f} "
            f"(Schmidt {lit['Schmidt']:,.0f}, Soufi {lit['Soufi']:,.0f}, "
            f"Mori {lit['Mori']:,.0f}, Li {lit['Li']:,.0f}) = {m['parca_vs_lit']:.2f}x."),
        "dnag-not-maintained-across-generations": (
            sim["frac_zero"],
            f"DnaG is at zero for {sim['frac_zero']:.1%} of {sim['n_timesteps']:,} "
            f"timesteps across {sim['n_files']} seeds; median {sim['median']:.0f}, "
            f"mean {sim['mean']:.2f}, max {sim['max']:.0f}."),
        "supply-below-gate-demand": (
            m["frac_below_gate_demand"],
            f"Below the gate's demand of {m['gate_demand_per_oric']} copies per "
            f"origin for {m['frac_below_gate_demand']:.1%} of the lineage."),
    }


def main() -> int:
    from ruamel.yaml import YAML
    from viva_superpowers import study_io
    from viva_superpowers.post_sim import StudyContext, build_report, write_report
    from viva_superpowers.study_status import derive_status
    from viva_superpowers.study_verdict import severity_gate, write_gate_evaluator
    from v2ecoli.library import dnag_deficit as dd

    print("building report card ...")
    r = subprocess.run([sys.executable, str(REPO / "scripts/study_report_cards.py"),
                        "--study", STUDY, "--card", CARD],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:], file=sys.stderr)
        return 1

    ctx = StudyContext.load(REPO, STUDY)
    cards = {p.name[: -len(".verdict.json")]: json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((STUDY_DIR / "viz" / "report_card").glob("*.verdict.json"))}
    report = build_report(STUDY, CANONICAL_RUN, cards)
    report["gate"] = severity_gate(report)
    write_report(ctx, report)
    card_gate = report["gate"]["status"]
    print(f"report.json - card gate: {card_gate} "
          f"({report['counts']['hard_mismatch']} hard mismatch)")

    ryaml = YAML(); ryaml.preserve_quotes = True; ryaml.width = 4096
    spec_path = STUDY_DIR / "study.yaml"
    spec = ryaml.load(spec_path.read_text(encoding="utf-8"))
    cfg = spec["report_card_refs"][CARD]

    def _p(k):
        v = cfg[k]
        return v if str(v).startswith("/") else str(REPO / v)

    m = dd.measure(_p("cache_dir"), _p("bundle_glob"), _p("proteome_script"),
                   fixture=_p("fixture"))
    measured = measure_tests(m)

    outcomes = {}
    for t in spec.get("behavior_tests") or []:
        name = t.get("name")
        if name not in measured:
            continue
        val, detail = measured[name]
        pi = dict(t.get("pass_if") or {})
        outcomes[name] = {
            "result": "PASS" if _apply(val, pi) else "FAIL",
            "measured_value": round(val, 6) if isinstance(val, float) else val,
            "evaluated_by": "agent",
            "operator": f"derived/{pi.get('op')} {pi.get('value')}",
            "gate_class": t.get("gate_class"),
            "detail": detail,
        }
    outcomes[CARD] = {
        "result": "PASS" if card_gate == "pass" else "FAIL",
        "measured_value": card_gate, "evaluated_by": "code",
        "operator": "report_card/severity_gate",
        "detail": (f"{report['counts']['within_tol']}/{report['counts']['axes']} axes "
                   f"within tolerance; {report['counts']['hard_mismatch']} hard mismatch."),
    }

    # This study runs no simulation, so there is no runs.db row to attach to.
    # Record a single analysis run so the gate has a canonical run to read.
    runs = spec.get("runs")
    if not isinstance(runs, list):
        runs = []
    entry = next((r for r in runs if r.get("name") == CANONICAL_RUN), None)
    if entry is None:
        from ruamel.yaml.comments import CommentedMap
        entry = CommentedMap(); entry["name"] = CANONICAL_RUN
        runs.append(entry)
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry["status"] = "complete"
    entry["canonical"] = True
    entry.setdefault("started_at", now)
    entry.setdefault("started_at_iso", now)
    entry["kind"] = "analysis"
    entry["note"] = ("Deterministic analysis of the ParCa cache, the proteomics "
                     "datasets, and the reused permissive lineage from "
                     "replisome-gate-sufficiency. No new simulation.")
    entry["outcomes"] = outcomes
    entry["computed_outcomes"] = {k: dict(v) for k, v in outcomes.items()}
    spec["runs"] = runs

    derived = derive_status(spec, list(runs), has_verdicts=True)
    for axis, info in derived.items():
        spec[axis] = info["value"]
    for axis, val in AUTHORED_AXES.items():
        spec.setdefault(axis, val)
    n_fail = sum(1 for o in outcomes.values() if o["result"] == "FAIL")
    spec["gate_status"] = "failed" if n_fail else "passed"

    buf = StringIO(); ryaml.dump(spec, buf)
    study_io.atomic_write(spec_path, buf.getvalue())
    write_gate_evaluator(STUDY_DIR)

    print(f"\noutcomes ({CANONICAL_RUN}):")
    for k, o in outcomes.items():
        print(f"  {o['result']:4} [{str(o.get('gate_class') or 'code')[:20]:20}] {k} = {o['measured_value']}")
    print(f"\ngate_status: {spec['gate_status']}  bottleneck: {m['bottleneck_step']} "
          f"({m['bottleneck_percentile']:.1f} pct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
