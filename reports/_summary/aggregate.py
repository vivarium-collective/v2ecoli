"""Filesystem -> InvestigationSummary dict. No HTML, no sims, read-only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_UNGRADED = "ungraded"


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _canonical_result(study: dict) -> str | None:
    for run in study.get("runs", []) or []:
        if run.get("canonical"):
            return run.get("result")
    return None


def _first_finding(study: dict) -> str | None:
    for f in study.get("findings", []) or []:
        stmt = f.get("statement")
        if stmt:
            return " ".join(str(stmt).split())  # collapse folded-yaml whitespace
    return None


def _card_name(html_ref: str) -> str:
    return Path(html_ref).stem  # "viz/report_card/standard.html" -> "standard"


def _verdict_path(study_dir: Path, html_ref: str) -> Path:
    return study_dir / html_ref.replace(".html", ".verdict.json")


def _card_stub(study_dir: Path, html_ref: str) -> dict[str, Any]:
    name = _card_name(html_ref)
    vpath = _verdict_path(study_dir, html_ref)
    hpath = study_dir / html_ref
    overall = None
    missing = not hpath.exists()
    if vpath.exists():
        try:
            overall = json.loads(vpath.read_text()).get("overall")
        except (json.JSONDecodeError, OSError):
            overall = None
    else:
        missing = True

    html = ""
    if hpath.exists():
        try:
            html = hpath.read_text()
        except OSError:
            html = ""
            missing = True

    _low = html.lower()
    # Route any card carrying page-level markup (full doc OR a bare
    # <style>/<script>/<link> block) through the iframe-isolation path so a
    # fragment's styles/scripts can't bleed into the summary page.
    is_full_doc = any(tok in _low for tok in ("<html", "<style", "<script", "<link"))

    return {
        "name": name,
        "overall": overall,
        "graded": overall not in (None, _UNGRADED),
        "html": html,
        "is_full_doc": is_full_doc,
        "axes": [],
        "missing": missing,
    }


def _graded_axes(study_dir: Path, html_ref: str) -> list[dict[str, Any]]:
    """Flattened axes across graded groups of one card's verdict.json."""
    vpath = _verdict_path(study_dir, html_ref)
    if not vpath.exists():
        return []
    try:
        data = json.loads(vpath.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("overall") in (None, _UNGRADED):
        return []
    axes: list[dict[str, Any]] = []
    for group in (data.get("groups") or {}).values():
        for a in group.get("axes", []) or []:
            axes.append({
                "label": a.get("label"),
                "verdict": a.get("verdict"),
                "value": a.get("value"),
                "meter": a.get("meter"),
            })
    return axes


def _config_json(study: dict) -> dict[str, Any]:
    """The study's real baseline config: the composite that drives the run
    (from conditions.baseline) plus the comparison run settings.

    Only non-null keys are included, so studies that omit a field (e.g. a
    ParCa study with no comparison block) get a clean object rather than a
    spray of nulls.
    """
    baseline = (study.get("conditions", {}) or {}).get("baseline", {}) or {}
    comparison = study.get("comparison", {}) or {}
    cfg: dict[str, Any] = {}
    if baseline.get("composite"):
        cfg["composite"] = baseline["composite"]
    if baseline.get("params"):
        cfg["params"] = baseline["params"]
    if comparison.get("seeds") is not None:
        cfg["seeds"] = comparison["seeds"]
    if comparison.get("generations") is not None:
        cfg["generations"] = comparison["generations"]
    return cfg


def aggregate(slug: str, workspace_root: str | Path) -> dict[str, Any]:
    ws = Path(workspace_root)
    inv_dir = ws / "investigations" / slug
    inv = _load_yaml(inv_dir / "investigation.yaml")

    studies: list[dict[str, Any]] = []
    rollup = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for study_slug in inv.get("studies", []) or []:
        study_dir = inv_dir / "studies" / study_slug
        study = _load_yaml(study_dir / "study.yaml")
        result = _canonical_result(study)
        if result in rollup:
            rollup[result] += 1
        cards = []
        for ref in study.get("report_cards", []) or []:
            card = _card_stub(study_dir, ref)
            card["axes"] = _graded_axes(study_dir, ref)
            cards.append(card)
        studies.append({
            "slug": study_slug,
            "title": study.get("title") or study.get("name") or study_slug,
            "status": study.get("status"),
            "result": result,
            "prerequisites": (study.get("pipeline_gate", {}) or {}).get("prerequisites", []) or [],
            "finding": _first_finding(study),
            "cards": cards,
            "config_json": _config_json(study),
        })

    # Build the verdict matrix
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    for s in studies:
        cells: dict[str, str | None] = {}
        for card in s["cards"]:
            for axis in card["axes"]:
                label = axis["label"]
                if label and label not in columns:
                    columns.append(label)
                if label:
                    cells[label] = axis["verdict"]
        rows.append({"study": s["slug"], "cells": cells})
    matrix = {"columns": columns, "rows": rows}

    return {
        "slug": slug,
        "title": inv.get("title") or slug,
        "question": inv.get("question") or "",
        "studies": studies,
        "rollup": rollup,
        "matrix": matrix,
    }
