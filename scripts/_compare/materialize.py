"""Materialize a study's report_cards + behavior_tests from its `comparison.cards`.

You author only `comparison.cards`; the CLI calls this on run to (re)write the
gating fields into the same study.yaml — one `report_card_axis` behavior_test per
GRADED card (standard/statistical; config/parca render but don't gate), pointing
at the canonical per-study card dir. Every other key (narrative, comparison
block, pipeline_gate, …) is preserved. Idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts._compare.study_spec import StudySpec, REPO

# report_card_axis verdict -> dashboard run-outcome (UPPERCASE; drift = PARTIAL
# so the pill carries the "within tolerance, with drift" caveat).
_OUTCOME = {"within_tol": "PASS", "drift": "PARTIAL", "mismatch": "FAIL",
            "ungraded": "PENDING"}


def card_root(spec: StudySpec) -> str:
    """The per-investigation card root the verdict/behavior_tests address."""
    return f"docs/report_cards/{spec.invest_name}"


def materialized_fields(spec: StudySpec) -> dict:
    """report_cards (viz embeds) + a modular `tests` list of report_card modules
    (one per assigned card). Graded cards carry a report_card_axis measure so the
    gate aggregates; config/parca are informational (no measure)."""
    cdir = f"{card_root(spec)}/{spec.name}"          # docs/report_cards/<invest>/<name>
    tests = []
    for c in spec.cards:
        t = {"name": f"{c}-vs-vecoli", "kind": "report_card", "card": c,
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {spec.name} ({c} card)?"}
        if c in spec.graded_cards:
            t["measure"] = {"kind": "report_card_axis", "card": cdir, "group": c}
        tests.append(t)
    return {
        "report_cards": [f"viz/report_card/{c}.html" for c in spec.cards],
        "tests": tests,
    }


def materialize_study(spec: StudySpec) -> Path:
    """Rewrite the study.yaml's report_cards + the modular `tests` list from its cards,
    preserving every other key; ensure an independent pipeline_gate. Returns the
    study path."""
    path = Path(spec.study_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(materialized_fields(spec))
    data.pop("behavior_tests", None)   # replaced by the modular `tests` list
    # Pipeline DAG: the study's authored `depends_on` (a list of prerequisite
    # study names) becomes pipeline_gate.prerequisites — the prerequisites must
    # pass before this study's gate is evaluated (parca gates the per-condition
    # studies; the single-seed basal gates the statistical study).
    data["pipeline_gate"] = {"prerequisites": list(data.get("depends_on") or []),
                             "enables": []}
    # Canonical run + per-test outcomes from the study's verdict JSON (when it
    # exists) so the dashboard pill strip shows the REAL result per card.
    run = {"name": f"{spec.name}-comparison", "kind": "analysis", "canonical": True,
           "description": f"v2e-compare study {spec.name}"}
    vpath = Path(REPO) / card_root(spec) / spec.name / "report_card_verdict.json"
    if vpath.is_file():
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
        groups = verdict.get("groups") or {}
        outcomes = {}
        for bt in data.get("tests") or []:
            grp = (bt.get("measure") or {}).get("group")
            if grp is None:
                continue
            gv = (groups.get(grp) or {}).get("verdict", "ungraded")
            outcomes[bt["name"]] = {"result": _OUTCOME.get(gv, "PENDING"),
                                    "detail": f"report card '{grp}': {gv}"}
        if outcomes:
            run.update(status="completed",
                       result=_OUTCOME.get(verdict.get("overall", "ungraded"), "PENDING"),
                       outcomes=outcomes)
            data["status"] = "evaluated"
    data["runs"] = [run]
    # Declare the v2ecoli baseline under a top-level `conditions:` block (the
    # v2ecoli baseline composite for this study's biological condition). The
    # `conditions:` block is what signals the dashboard's v4-REDESIGN study path,
    # which PRESERVES a list `tests:` (our report_card modules). A flat
    # `baseline: [list]` instead routes to the legacy-v4 path, which expects
    # `tests:` to be a dict and silently drops our module list.
    data.pop("baseline", None)
    data.setdefault("conditions", {
        "baseline": {
            "composite": "v2ecoli.composites.baseline.baseline",
            "params": {"condition": spec.condition},
        },
    })
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
