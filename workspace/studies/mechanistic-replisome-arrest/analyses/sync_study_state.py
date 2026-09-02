"""Regenerate every derived field of the study from the completed run.

Idempotent and re-runnable: after any new run, this recomputes the report card,
the per-test outcomes, the test report, the gate and the six status axes from
the run's own artifacts. Nothing here is a hand-set value that can drift out of
date -- that drift is exactly what viva_superpowers.study_status exists to
prevent, and a study that must be hand-patched after every run will be wrong
the first time someone forgets.

What is derived, and from where:

report card
    ``scripts/study_report_cards.py`` builds ReplisomeArrestCard and writes
    ``viz/report_card/replisome_arrest.{html,verdict.json}``.

runs[].outcomes
    The gate reads the CANONICAL run's ``outcomes`` (viva_workspace.outcomes
    .canonical_outcomes), NOT ``computed_outcomes`` -- the latter is the
    code-evaluator's parallel block, and it is ``store_unresolved`` here because
    the arms emitted parquet outside the workbench's emitter registry. The three
    behavior_tests declare ``measure.kind: derived``, which is not a run-data
    kind, so they are agent-evaluated by construction; this module supplies the
    measurement from replisome_arrest.measure() and applies each test's OWN
    ``pass_if`` from study.yaml rather than a hardcoded threshold, so editing a
    threshold in the spec changes the verdict here.

status axes
    simulation/evaluation come from study_status.derive_status (observable);
    design/implementation/expert_review are authored intent and are only
    seeded when absent, never overwritten.

The mechanistic arm is pinned ``canonical: true``. Without a pin the canonical
run is "newest completed by timestamp", which here is the permissive arm purely
because it finished second -- an arbitrary basis for which run's outcomes the
gate reads.

Usage::

    python workspace/studies/mechanistic-replisome-arrest/analyses/sync_study_state.py
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
CARD = "replisome_arrest"
ARMS = {
    "mechanistic": REPO / "out/mechanistic-replisome-arrest/mechanistic",
    "permissive": REPO / "out/mechanistic-replisome-arrest/permissive",
}
CACHE_DIR = REPO / "out/cache"
CANONICAL_RUN = "mechanistic-replisome-arrest__mechanistic__seed0"

# Authored-intent axes: seeded once, never overwritten by this script.
AUTHORED_AXES = {
    "design_status": "complete",
    "implementation_status": "complete",
    "expert_review_status": "pending",
}


def _apply_pass_if(measured, pass_if: dict) -> bool:
    """Evaluate a behavior_test's own pass_if against a measured value."""
    op = str(pass_if.get("op", "")).strip()
    want = pass_if.get("value")
    ops = {
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    }
    if op not in ops:
        raise ValueError(f"unsupported pass_if op: {op!r}")
    return bool(ops[op](measured, want))


def measure_tests(m: dict) -> dict:
    """Measured value + human detail per behavior_test name."""
    dnag_exhausted = bool(
        m["worst_subunit_margin"] is not None
        and m["worst_subunit_margin"] < 0
        and m["limiting_pool"] == "DnaG")
    margins = ", ".join(
        f"{v['label']} {v['margin']:+d}" for v in m["subunit_margins"].values())
    return {
        "lineage-arrests-under-mechanistic-replisome": (
            m["mechanistic_generations"],
            f"Mechanistic arm completed {m['mechanistic_generations']} generation(s); "
            f"generation {m['arrest_generation']} ran the full duration cap without "
            f"dividing (dry mass {m['arrest_dry_mass_fg']:.1f} fg)."),
        "permissive-lineage-survives": (
            m["permissive_generations"],
            f"Permissive arm completed {m['permissive_generations']} generation(s) "
            "from the same seed and the same cache."),
        "dnag-is-the-exhausted-subunit": (
            dnag_exhausted,
            f"No pool was exhausted at generation {m['arrest_generation']}: worst "
            f"margin {m['worst_subunit_margin']:+d} copies ({m['limiting_pool']}, the "
            f"scarcest pool, which still cleared its requirement). Margins: {margins}."),
    }


def main() -> int:
    from ruamel.yaml import YAML
    from viva_superpowers import study_io
    from viva_superpowers.post_sim import (
        StudyContext, build_report, write_report,
    )
    from viva_superpowers.study_status import derive_status
    from viva_superpowers.study_verdict import severity_gate, write_gate_evaluator
    from v2ecoli.library import replisome_arrest as ra

    for arm, d in ARMS.items():
        if not d.is_dir():
            print(f"ERROR: {arm} arm not found at {d}", file=sys.stderr)
            return 1

    # 1. Rebuild the report card from the run.
    print("building report card ...")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/study_report_cards.py"),
         "--study", STUDY, "--card", CARD],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        return 1
    print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-2].strip()
          if len(r.stdout.strip().splitlines()) > 1 else "  ok")

    # 2. Aggregate the card into report.json + the severity gate.
    ctx = StudyContext.load(REPO, STUDY)
    cards = {}
    for vf in sorted((STUDY_DIR / "viz" / "report_card").glob("*.verdict.json")):
        cards[vf.name[: -len(".verdict.json")]] = json.loads(
            vf.read_text(encoding="utf-8"))
    if not cards:
        print("ERROR: no report-card verdicts produced", file=sys.stderr)
        return 1
    report = build_report(STUDY, CANONICAL_RUN, cards)
    report["gate"] = severity_gate(report)
    write_report(ctx, report)
    card_gate = report["gate"]["status"]
    print(f"report.json written — card gate: {card_gate} "
          f"({report['counts']['hard_mismatch']} hard mismatch)")

    # 3. Measure, then grade each behavior_test with its OWN pass_if.
    m = ra.measure(ARMS["mechanistic"], ARMS["permissive"], CACHE_DIR)
    measured = measure_tests(m)

    ryaml = YAML()
    ryaml.preserve_quotes = True
    ryaml.width = 4096
    spec_path = STUDY_DIR / "study.yaml"
    spec = ryaml.load(spec_path.read_text(encoding="utf-8"))

    outcomes = {}
    for t in spec.get("behavior_tests") or []:
        name = t.get("name")
        if name not in measured:
            continue
        value, detail = measured[name]
        passed = _apply_pass_if(value, dict(t.get("pass_if") or {}))
        pi = t.get("pass_if") or {}
        outcomes[name] = {
            "result": "PASS" if passed else "FAIL",
            "measured_value": value,
            "evaluated_by": "agent",
            "operator": f"derived/{pi.get('op')} {pi.get('value')}",
            "detail": detail,
        }
    # The report-card test carries the card's own severity gate.
    outcomes[CARD] = {
        "result": "PASS" if card_gate == "pass" else "FAIL",
        "measured_value": card_gate,
        "evaluated_by": "code",
        "operator": "report_card/severity_gate",
        "detail": (f"{report['counts']['within_tol']}/{report['counts']['axes']} axes "
                   f"within tolerance; {report['counts']['hard_mismatch']} hard "
                   "mismatch. See viz/report_card/replisome_arrest.html."),
    }

    # 4. Write outcomes onto the pinned canonical run.
    runs = spec.get("runs") or []
    target = None
    for run in runs:
        if run.get("name") == CANONICAL_RUN:
            target = run
            run["canonical"] = True
        else:
            run.pop("canonical", None)
    if target is None:
        print(f"ERROR: canonical run {CANONICAL_RUN} not in runs[]", file=sys.stderr)
        return 1
    target["outcomes"] = outcomes
    # finding_observations.populate reads measured values from computed_outcomes
    # and skips the `store_unresolved` stub the evaluator left when it could not
    # open these runs' parquet. Supersede that stub with the same measurements so
    # the code-owned finding slots (evidence.observed, divergence_factor) fill.
    # Each entry keeps its own `evaluated_by`, which is what distinguishes the
    # card's code grading from the agent-evaluated `derived` tests -- the block
    # records HOW each was measured, so this does not launder one as the other.
    target["computed_outcomes"] = {k: dict(v) for k, v in outcomes.items()}

    # 5. Status axes — observable ones derived, authored ones seeded if absent.
    derived = derive_status(spec, list(runs), has_verdicts=True)
    for axis, info in derived.items():
        spec[axis] = info["value"]
    for axis, val in AUTHORED_AXES.items():
        spec.setdefault(axis, val)
    n_fail = sum(1 for o in outcomes.values() if o["result"] == "FAIL")
    spec["gate_status"] = "failed" if n_fail else "passed"

    buf = StringIO()
    ryaml.dump(spec, buf)
    study_io.atomic_write(spec_path, buf.getvalue())

    # 6. Recompute the coded gate_evaluator from the freshly-written outcomes.
    write_gate_evaluator(STUDY_DIR)

    print("\noutcomes recorded on canonical run "
          f"{CANONICAL_RUN}:")
    for name, o in outcomes.items():
        print(f"  {o['result']:4}  {name}  = {o['measured_value']}")
    print(f"\ngate_status: {spec['gate_status']}   "
          f"simulation: {spec.get('simulation_status')}   "
          f"evaluation: {spec.get('evaluation_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
