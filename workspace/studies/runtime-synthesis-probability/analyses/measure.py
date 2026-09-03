"""Record the sweep's results into the study: outcomes, gate, status axes.

Same pattern as the sync scripts in the preceding studies and idempotent for the
same reason. Each behavior_test is graded with its OWN pass_if from study.yaml,
so a threshold edited in the spec changes the verdict here and nowhere else --
which also means a failing threshold cannot be quietly relaxed in code.

Usage::

    python workspace/studies/runtime-synthesis-probability/analyses/measure.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
STUDY = STUDY_DIR.name
CARD = "runtime_synth_prob"
CANONICAL_RUN = f"{STUDY}__sweep"
AUTHORED = {"design_status": "complete", "implementation_status": "complete",
            "expert_review_status": "pending"}


def _apply(v, pi: dict) -> bool:
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    return bool(ops[str(pi.get("op")).strip()](v, pi.get("value")))


def measure_tests(m: dict) -> dict:
    return {
        "runtime-probability-is-depressed": (
            m["actual_depression"],
            f"Probability in force: cohort median {m['cohort_actual_median']:.3e} vs "
            f"comparison {m['comparison_actual_median']:.3e} = "
            f"{m['actual_depression']:.4f} ({1/m['actual_depression']:.0f}x depressed)."),
        "realized-tracks-runtime-probability": (
            m["selection_fairness"],
            f"Realized per unit of in-force probability: cohort "
            f"{m['cohort_realized_per_prob']:.3e} vs comparison "
            f"{m['comparison_realized_per_prob']:.3e} = {m['selection_fairness']:.3f}."),
        "actual-tracks-target": (
            m["cohort_actual_over_target"],
            f"Cohort actual/target = {m['cohort_actual_over_target']:.3f} "
            f"(comparison {m['comparison_actual_over_target']:.3f}); TF bound "
            f"cohort {m['cohort_tf_median']:.2f}, comparison "
            f"{m['comparison_tf_median']:.2f}."),
        "comparison-group-behaves-normally": (
            m["comparison_control_fold"],
            f"Comparison actual/target agreement fold = "
            f"{m['comparison_control_fold']:.2f} (1.00 = identical)."),
        "cohort-remains-depressed": (
            m["cohort_max_realized_over_assigned"],
            f"Cohort max realized/assigned = "
            f"{m['cohort_max_realized_over_assigned']:.4f}."),
    }


def main() -> int:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    from viva_superpowers import study_io
    from viva_superpowers.post_sim import StudyContext, build_report, write_report
    from viva_superpowers.study_status import derive_status
    from viva_superpowers.study_verdict import severity_gate, write_gate_evaluator
    from v2ecoli.library import runtime_synth_prob as st

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
    gate = report["gate"]["status"]
    print(f"report.json - card gate: {gate} "
          f"({report['counts']['hard_mismatch']} hard mismatch)")

    ryaml = YAML(); ryaml.preserve_quotes = True; ryaml.width = 4096
    sp = STUDY_DIR / "study.yaml"
    spec = ryaml.load(sp.read_text(encoding="utf-8"))
    cfg = spec["report_card_refs"][CARD]

    def _p(k):
        v = cfg[k]
        return v if str(v).startswith("/") else str(REPO / v)

    m = st.measure(_p("cache_dir"), _p("out_root"))
    measured = measure_tests(m)

    outcomes = {}
    for t in spec.get("behavior_tests") or []:
        name = t.get("name")
        if name not in measured:
            continue
        val, detail = measured[name]
        pi = dict(t.get("pass_if") or {})
        if val is None:
            # Not machine-gradable: no binary attribute was found, so there is no
            # number to compare. Recorded as FAIL with the reasoning rather than
            # silently omitted.
            res = "FAIL"
        else:
            res = "PASS" if _apply(val, pi) else "FAIL"
        outcomes[name] = {
            "result": res,
            "measured_value": (round(val, 6) if isinstance(val, float) else val),
            "evaluated_by": "agent",
            "operator": f"derived/{pi.get('op')} {pi.get('value')}",
            "gate_class": t.get("gate_class"),
            "detail": detail,
        }
    outcomes[CARD] = {
        "result": "PASS" if gate == "pass" else "FAIL",
        "measured_value": gate, "evaluated_by": "code",
        "operator": "report_card/severity_gate",
        "detail": (f"{report['counts']['within_tol']}/{report['counts']['axes']} axes "
                   f"within tolerance; {report['counts']['hard_mismatch']} hard mismatch."),
    }

    runs = spec.get("runs") if isinstance(spec.get("runs"), list) else []
    entry = next((r for r in runs if r.get("name") == CANONICAL_RUN), None)
    if entry is None:
        entry = CommentedMap(); entry["name"] = CANONICAL_RUN; runs.append(entry)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry["status"] = "complete"; entry["canonical"] = True
    entry.setdefault("started_at", now); entry.setdefault("started_at_iso", now)
    entry["note"] = ("Analysis only: no new simulation. Reuses the 3 seed x 3 "
                     "generation sweep from silent-transcription-units.")
    entry["outcomes"] = outcomes
    entry["computed_outcomes"] = {k: dict(v) for k, v in outcomes.items()}
    spec["runs"] = runs

    for axis, info in derive_status(spec, list(runs), has_verdicts=True).items():
        spec[axis] = info["value"]
    for axis, val in AUTHORED.items():
        spec.setdefault(axis, val)
    n_fail = sum(1 for o in outcomes.values() if o["result"] == "FAIL")
    spec["gate_status"] = "failed" if n_fail else "passed"

    buf = StringIO(); ryaml.dump(spec, buf); study_io.atomic_write(sp, buf.getvalue())
    write_gate_evaluator(STUDY_DIR)

    print(f"\noutcomes ({CANONICAL_RUN}):")
    for k, o in outcomes.items():
        print(f"  {o['result']:4} [{str(o.get('gate_class') or 'code')[:20]:20}] "
              f"{k} = {o['measured_value']}")
    print(f"\ngate_status: {spec['gate_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
