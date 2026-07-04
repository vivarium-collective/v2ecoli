# v2ecoli/workflow/report_cards/__init__.py
"""Report cards: concrete ``ReportCardStep`` subclasses + their runner plumbing.

The ``ReportCardStep`` base Step now lives in ``v2ecoli/workflow/post_sim.py``
alongside the ``Visualization`` base (and ``Analysis`` in
``v2ecoli/workflow/analysis.py``); it is re-exported here for the concrete cards
and the runner. A report card emits a rendered ``view`` (the card HTML) plus
``data`` (the verdict_json map). Unlike ``Analysis`` — which consumes a live
DuckDB sim-output connection — a report card's input is a ``StudyContext`` (the
study's spec + dir), so cards grade run-free. Subclasses that set ``name``
auto-register in ``REPORT_CARD_REGISTRY``.

The runner (``scripts/study_report_cards.py``) builds a ``bigraph_schema`` core,
instantiates each registered card, calls ``applies``/``build``, and writes the
``view`` → ``viz/report_card/<name>.html`` and ``data`` → ``<name>.verdict.json``
(the files the dashboard discovers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from v2ecoli.workflow.post_sim import REPORT_CARD_REGISTRY, ReportCardStep  # noqa: F401


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats with None, recursively (bundle JSON.parse safe)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@dataclass
class StudyContext:
    study_name: str
    study_dir: Path
    spec: dict
    ws_root: Path

    @classmethod
    def load(cls, ws_root: Path, study_name: str) -> "StudyContext":
        sd = ws_root / "workspace" / "studies" / study_name
        spec_path = sd / "study.yaml"
        spec = {}
        if spec_path.is_file():
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        return cls(study_name=study_name, study_dir=sd, spec=spec, ws_root=ws_root)

    def run_zarr_paths(self) -> list[Path]:
        return sorted(self.study_dir.glob("runs.*.zarr"))

    @property
    def card_dir(self) -> Path:
        return self.study_dir / "viz" / "report_card"


def write_card(ctx: StudyContext, name: str, verdict: dict, html: str) -> Path:
    """Write <card>.html + <card>.verdict.json into the study's report_card dir.
    Returns the html path. Verdict is sanitized + written with allow_nan=False."""
    d = ctx.card_dir
    d.mkdir(parents=True, exist_ok=True)
    html_path = d / f"{name}.html"
    html_path.write_text(html, encoding="utf-8")
    (d / f"{name}.verdict.json").write_text(
        json.dumps(_sanitize(verdict), indent=1, allow_nan=False) + "\n",
        encoding="utf-8")
    return html_path


def prune(ctx: StudyContext, keep: set[str]) -> list[str]:
    """Delete <card>.html (+ sibling .verdict.json) under the study's report_card
    dir whose stem is not in `keep`. Returns pruned stems. Touches only that dir."""
    d = ctx.card_dir
    pruned: list[str] = []
    if not d.is_dir():
        return pruned
    for html in sorted(d.glob("*.html")):
        stem = html.name[: -len(".html")]
        if stem not in keep:
            html.unlink()
            vf = html.with_name(stem + ".verdict.json")
            if vf.is_file():
                vf.unlink()
            pruned.append(stem)
    return pruned


def applicable(ctx: StudyContext, core, only: "str | None" = None) -> list:
    """Instantiated report-card Steps to emit for a study. If the study spec lists
    `report_cards:`, only those names are eligible; otherwise every registered card
    is eligible. A card is emitted when eligible AND its applies(ctx) is True.
    `only` (a name, or None/'all') narrows to a single card. `core` is a
    bigraph-schema core (built once by the caller) used to instantiate Steps."""
    declared = ctx.spec.get("report_cards")
    want = None if (only in (None, "all")) else {only}
    out = []
    for nm, cls in REPORT_CARD_REGISTRY.items():
        if want is not None and nm not in want:
            continue
        if declared is not None and nm not in declared:
            continue
        try:
            step = cls({}, core=core)
            if step.applies(ctx):
                out.append(step)
        except Exception:  # noqa: BLE001 — one broken card never aborts selection
            continue
    return out


# Register built-in cards (import for side effect). These modules all exist, so
# import unconditionally — a real import error must surface, not be masked.
from . import tests_card  # noqa: E402,F401
from . import vs_literature_card  # noqa: E402,F401
from . import vs_vecoli_card  # noqa: E402,F401
