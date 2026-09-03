"""Regenerate every derived field of the study from the completed sweep.

Idempotent and re-runnable, and the same pattern as study 1's
sync_study_state.py: after any new run this recomputes the report card, the
per-test outcomes, the test report, the gate and the six status axes from the
run's own artifacts. Nothing here is a hand-set value that can drift.

Reads the distilled evidence bundles, not the raw parquet -- the bulk history
was deleted at close-out after analyses/distill_evidence.py --verify confirmed
the margins reproduce exactly.

Each behavior_test is graded with its OWN pass_if from study.yaml rather than a
hardcoded threshold, so editing a threshold in the spec changes the verdict
here. The three-arm design means several tests are cross-arm (paired relief) or
cross-seed (spread), which is why they are `measure.kind: derived` and
agent-evaluated: they are not single-run listener reads.

Usage::

    python workspace/studies/replisome-gate-sufficiency/analyses/sync_study_state.py
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
CARD = "gate_sufficiency"
OUT_ROOT = REPO / "out" / STUDY
BUNDLE_DIR = STUDY_DIR / "evidence"
CANONICAL_RUN = f"{STUDY}__mechanistic__seed0"

AUTHORED_AXES = {
    "design_status": "complete",
    "implementation_status": "complete",
    "expert_review_status": "pending",
}


def _apply_pass_if(measured, pass_if: dict) -> bool:
    op = str(pass_if.get("op", "")).strip()
    want = pass_if.get("value")
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    if op not in ops:
        raise ValueError(f"unsupported pass_if op: {op!r}")
    return bool(ops[op](measured, want))


def measure_tests(m: dict) -> dict:
    """Measured value + human detail per behavior_test name."""
    mech, perm, abl = (m["arms"]["mechanistic"], m["arms"]["permissive"],
                       m["arms"]["dnag-ablation"])
    worst = ", ".join(f"seed {k} {v['pool']} {v['margin']:+d}"
                      for k, v in sorted(m["per_seed_worst"].items()))
    relief = ", ".join(f"seed {k} {v:+d}" for k, v in sorted(m["ablation_relief"].items()))
    return {
        "lineage-stalls-under-sufficiency-gate": (
            mech["divided_max"],
            f"{mech['n_stalled']}/{mech['n_seeds']} mechanistic seeds stalled; "
            f"maximum divided-generation count {mech['divided_max']} of "
            f"{m['expected_generations']}. Stall generations "
            f"{mech['stall_generations']}."),
        "stall-generation-is-seed-consistent": (
            mech["stall_spread"],
            f"Stall generation {min(mech['stall_generations'])}-"
            f"{max(mech['stall_generations'])} across {mech['n_seeds']} seeds; "
            f"spread {mech['stall_spread']}."),
        "some-pool-is-short-at-the-stall": (
            m["worst_margin"],
            f"Worst margin {m['worst_margin']:+d} copies. Per seed: {worst}."),
        "dnag-is-the-limiting-pool": (
            m["limiting_pool_unanimous"],
            f"Worst-margin pool is {m['limiting_pool']} in "
            f"{'every' if m['limiting_pool_unanimous'] else 'some'} stalled seed "
            f"(pools seen: {', '.join(m['limiting_pools_seen'])})."),
        "permissive-completes-all-seeds": (
            perm["divided_min"],
            f"Permissive control divided {perm['divided_min']}-{perm['divided_max']} "
            f"generations across {perm['n_seeds']} seeds; {perm['n_stalled']} stalls."),
        "dropping-dnag-relieves-the-stall": (
            m["ablation_relief_min"],
            f"Ablation arm divided {abl['divided_min']}-{abl['divided_max']} vs "
            f"mechanistic {mech['divided_min']}-{mech['divided_max']}. Paired "
            f"relief: {relief}."),
    }


def main() -> int:
    from ruamel.yaml import YAML
    from viva_superpowers import study_io
    from viva_superpowers.post_sim import StudyContext, build_report, write_report
    from viva_superpowers.study_status import derive_status
    from viva_superpowers.study_verdict import severity_gate, write_gate_evaluator
    from v2ecoli.library import gate_sufficiency as gs

    if not BUNDLE_DIR.is_dir():
        print(f"ERROR: evidence bundle missing at {BUNDLE_DIR}", file=sys.stderr)
        return 1

    print("building report card ...")
    r = subprocess.run([sys.executable, str(REPO / "scripts/study_report_cards.py"),
                        "--study", STUDY, "--card", CARD],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        return 1

    ctx = StudyContext.load(REPO, STUDY)
    cards = {p.name[: -len(".verdict.json")]: json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((STUDY_DIR / "viz" / "report_card").glob("*.verdict.json"))}
    if not cards:
        print("ERROR: no report-card verdicts produced", file=sys.stderr)
        return 1
    report = build_report(STUDY, CANONICAL_RUN, cards)
    report["gate"] = severity_gate(report)
    write_report(ctx, report)
    card_gate = report["gate"]["status"]
    print(f"report.json written - card gate: {card_gate} "
          f"({report['counts']['hard_mismatch']} hard mismatch)")

    m = gs.measure(OUT_ROOT, BUNDLE_DIR)
    measured = measure_tests(m)

    ryaml = YAML(); ryaml.preserve_quotes = True; ryaml.width = 4096
    spec_path = STUDY_DIR / "study.yaml"
    spec = ryaml.load(spec_path.read_text(encoding="utf-8"))

    outcomes = {}
    for t in spec.get("behavior_tests") or []:
        name = t.get("name")
        if name not in measured:
            continue
        val, detail = measured[name]
        pi = t.get("pass_if") or {}
        outcomes[name] = {
            "result": "PASS" if _apply_pass_if(val, dict(pi)) else "FAIL",
            "measured_value": val,
            "evaluated_by": "agent",
            "operator": f"derived/{pi.get('op')} {pi.get('value')}",
            "detail": detail,
        }
    outcomes[CARD] = {
        "result": "PASS" if card_gate == "pass" else "FAIL",
        "measured_value": card_gate,
        "evaluated_by": "code",
        "operator": "report_card/severity_gate",
        "detail": (f"{report['counts']['within_tol']}/{report['counts']['axes']} axes "
                   f"within tolerance; {report['counts']['hard_mismatch']} hard "
                   "mismatch. See viz/report_card/gate_sufficiency.html."),
    }

    runs = spec.get("runs") or []
    target = None
    for run in runs:
        if run.get("name") == CANONICAL_RUN:
            target = run
            run["canonical"] = True
        else:
            run.pop("canonical", None)
    if target is None:
        print(f"ERROR: canonical run {CANONICAL_RUN} not in runs[]; "
              "run viva-sync-runs first", file=sys.stderr)
        return 1
    target["outcomes"] = outcomes
    target["computed_outcomes"] = {k: dict(v) for k, v in outcomes.items()}

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

    print(f"\noutcomes on canonical run {CANONICAL_RUN}:")
    for name, o in outcomes.items():
        print(f"  {o['result']:4}  {name} = {o['measured_value']}")
    print(f"\ngate_status: {spec['gate_status']}   "
          f"simulation: {spec.get('simulation_status')}   "
          f"evaluation: {spec.get('evaluation_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
