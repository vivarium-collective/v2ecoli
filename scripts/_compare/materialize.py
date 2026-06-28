"""Materialize a study's report_cards + behavior_tests from its `comparison.cards`.

You author only `comparison.cards`; the CLI calls this on run to (re)write the
gating fields into the same study.yaml — one `report_card_axis` behavior_test per
GRADED card (standard/statistical; config/parca render but don't gate), pointing
at the canonical per-study card dir. Every other key (narrative, comparison
block, pipeline_gate, …) is preserved. Idempotent.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from scripts._compare.study_spec import StudySpec


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
    data.setdefault("runs", [{"name": f"{spec.name}-comparison", "kind": "analysis",
                              "canonical": True,
                              "description": f"v2e-compare study {spec.name}"}])
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
