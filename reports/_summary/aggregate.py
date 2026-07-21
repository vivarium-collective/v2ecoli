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
    return {
        "name": name,
        "overall": overall,
        "graded": overall not in (None, _UNGRADED),
        "html": "",
        "is_full_doc": False,
        "axes": [],
        "missing": missing,
    }


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
        cards = [_card_stub(study_dir, ref) for ref in study.get("report_cards", []) or []]
        studies.append({
            "slug": study_slug,
            "title": study.get("title") or study.get("name") or study_slug,
            "status": study.get("status"),
            "result": result,
            "prerequisites": (study.get("pipeline_gate", {}) or {}).get("prerequisites", []) or [],
            "finding": _first_finding(study),
            "cards": cards,
        })

    return {
        "slug": slug,
        "title": inv.get("title") or slug,
        "question": inv.get("question") or "",
        "studies": studies,
        "rollup": rollup,
        "matrix": {"columns": [], "rows": []},
    }
