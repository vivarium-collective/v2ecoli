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
    """The report_cards + behavior_tests derived from a study's graded cards."""
    card_dir = f"{card_root(spec)}/{spec.name}"
    return {
        "report_cards": [f"{card_dir}/index.html"],
        "behavior_tests": [
            {"name": f"{c}-vs-vecoli",
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {spec.name} ({c} card)?",
             "measure": {"kind": "report_card_axis", "card": card_dir, "group": c}}
            for c in spec.graded_cards],
    }


def materialize_study(spec: StudySpec) -> Path:
    """Rewrite the study.yaml's report_cards + behavior_tests from its cards,
    preserving every other key; ensure an independent pipeline_gate. Returns the
    study path."""
    path = Path(spec.study_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(materialized_fields(spec))
    data.setdefault("pipeline_gate", {"prerequisites": [], "enables": []})
    # Canonical run + per-test outcomes from the study's verdict JSON (when it
    # exists) so the dashboard pill strip shows the REAL result per card.
    run = {"name": f"{spec.name}-comparison", "kind": "analysis", "canonical": True,
           "description": f"v2e-compare study {spec.name}"}
    vpath = Path(REPO) / card_root(spec) / spec.name / "report_card_verdict.json"
    if vpath.is_file():
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
        groups = verdict.get("groups") or {}
        outcomes = {}
        for bt in data.get("behavior_tests") or []:
            grp = (bt.get("measure") or {}).get("group")
            gv = (groups.get(grp) or {}).get("verdict", "ungraded")
            outcomes[bt["name"]] = {"result": _OUTCOME.get(gv, "PENDING"),
                                    "detail": f"report card '{grp}': {gv}"}
        if outcomes:
            run.update(status="completed",
                       result=_OUTCOME.get(verdict.get("overall", "ungraded"), "PENDING"),
                       outcomes=outcomes)
            data["status"] = "evaluated"
    data["runs"] = [run]
    # The dashboard's v3 study schema requires a non-empty baseline list of
    # composites; the v2ecoli baseline (for this study's biological condition)
    # is the runnable composite the comparison's v2ecoli side is built from.
    data.setdefault("baseline", [{
        "name": "v2ecoli-baseline",
        "composite": "v2ecoli.composites.baseline.baseline",
        "params": {"condition": spec.condition},
    }])
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
