"""Filesystem -> InvestigationSummary dict. No HTML, no sims, read-only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_UNGRADED = "ungraded"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding='utf-8') as fh:
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
            overall = json.loads(vpath.read_text(encoding='utf-8')).get("overall")
        except (json.JSONDecodeError, OSError):
            overall = None
    else:
        missing = True

    html = ""
    if hpath.exists():
        try:
            html = hpath.read_text(encoding='utf-8')
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
        data = json.loads(vpath.read_text(encoding='utf-8'))
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


def _prerequisites(study: dict) -> list[str]:
    """A study's ordering dependencies. Source of truth is `inputs.from`
    (Study Pipeline migration moved ordering off pipeline_gate.prerequisites,
    which the registry migrator now strips entirely). Falls back to the
    legacy `pipeline_gate.prerequisites` for any un-migrated study that still
    carries it."""
    from_inputs = [e.get("from") for e in (study.get("inputs") or []) if e.get("from")]
    if from_inputs:
        return from_inputs
    return (study.get("pipeline_gate", {}) or {}).get("prerequisites", []) or []


def _dag_order(studies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable topological sort by each study's in-set `prerequisites`: a study
    appears after every prerequisite that is itself in this list. Studies at the
    same DAG depth keep their input order. A cycle or unresolvable ref falls
    through in input order rather than dropping any study."""
    by_slug = {s["slug"]: s for s in studies}
    emitted: set[str] = set()
    order: list[dict[str, Any]] = []
    remaining = list(studies)
    while remaining:
        ready = [
            s for s in remaining
            if all(p in emitted for p in s["prerequisites"] if p in by_slug)
        ]
        if not ready:  # cycle / unresolvable — emit the rest as-is
            order.extend(remaining)
            break
        for s in ready:
            order.append(s)
            emitted.add(s["slug"])
            remaining.remove(s)
    return order


def aggregate(slug: str, workspace_root: str | Path) -> dict[str, Any]:
    ws = Path(workspace_root)
    inv_dir = ws / "investigations" / slug
    inv = _load_yaml(inv_dir / "investigation.yaml")

    studies: list[dict[str, Any]] = []
    rollup = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    # Registry model (Study Pipeline Spec 1): investigations REFERENCE top-level
    # studies/<slug>/ via `members:`. Legacy investigations used a nested
    # `studies:` list under investigations/<inv>/studies/<slug>/. The config-is-
    # the-unit model (post-Task-6) instead lists `comparison.configs[]`, with no
    # `members:`/`studies:` key at all -- resolve that case via the shared
    # study_spec loader (imported lazily: study_spec pulls in ReferenceEngine
    # etc., and reports/_summary/ should stay light for callers that only want
    # the legacy members: path). Both paths land on the same top-level
    # workspace/studies/<slug>/study.yaml.
    if (inv.get("comparison") or {}).get("configs"):
        from scripts._compare.study_spec import load_investigation
        _ctx, specs = load_investigation(inv_dir)
        member_slugs = [s.name for s in specs]
        # `comparison.configs[]` models per-condition runs (the config IS the
        # study), but this investigation also carries standalone root studies
        # -- `parca` (t=0 initial-state check) and `statistical` (multi-seed
        # Welch-t check) -- that are structurally not "one config, one study"
        # and so were deliberately left out of configs[] (see the Task 6
        # migration commit 4c759527). Those still declare this investigation
        # via their own `investigation:` back-reference, so pick them up by
        # scanning the top-level registry rather than hardcoding their names.
        known = set(member_slugs)
        for sp in sorted((ws / "studies").glob("*/study.yaml")):
            study_slug = sp.parent.name
            if study_slug in known:
                continue
            if (_load_yaml(sp) or {}).get("investigation") == slug:
                member_slugs.append(study_slug)
                known.add(study_slug)
    else:
        member_slugs = inv.get("members") or inv.get("studies") or []
    for study_slug in member_slugs:
        study_dir = ws / "studies" / study_slug
        if not (study_dir / "study.yaml").is_file():
            study_dir = inv_dir / "studies" / study_slug  # legacy nested fallback
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
            "prerequisites": _prerequisites(study),
            "finding": _first_finding(study),
            "cards": cards,
            "config_json": _config_json(study),
        })

    # Order studies by their prerequisite DAG (roots like `parca` first) so the
    # summary + matrix read in pipeline order regardless of the investigation's
    # members: list order — the registry migrator sorts members alphabetically,
    # which is not a valid pipeline order (a variant could precede its ParCa).
    studies = _dag_order(studies)

    # Build the verdict matrix
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    for s in studies:
        cells: dict[str, str | None] = {}
        for card in s["cards"]:
            # The per-observable matrix is built from the standard/parca cards
            # (which share the "cell mass (fg)" observable labels). The multi-seed
            # statistical card uses a different group structure + labels ("Cell
            # mass", plus ribosome/transcription axes) and is surfaced as its own
            # report card, not flattened into this grid.
            if card.get("name") == "statistical":
                continue
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
