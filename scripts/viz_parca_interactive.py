#!/usr/bin/env python
"""Interactive bigraph-viz2 view of the v2ecoli ParCa pipeline composite.

ParCa (the parameter calculator) is a 9-step process-bigraph pipeline.
`build_parca_document()` returns its structural spec (steps + wiring) WITHOUT
running the expensive fit, so this renders the pipeline structure cheaply.

Run:  .venv/bin/python scripts/viz_parca_interactive.py [--open]
Out:  out/viz/parca_interactive.html
"""
from __future__ import annotations

import argparse
import inspect
import sys
import webbrowser
from pathlib import Path

try:
    from bigraph_viz2 import emit_html
except ModuleNotFoundError:  # fall back to a sibling checkout of bigraph-viz2
    _sib = Path(__file__).resolve().parents[2] / "bigraph-viz2" / "py"
    if _sib.is_dir():
        sys.path.insert(0, str(_sib))
    from bigraph_viz2 import emit_html  # noqa: E402

from v2ecoli.processes.parca.composite import (  # noqa: E402
    build_parca_document, STEP_ORDER,
)
from v2ecoli.processes.parca.steps import ALL_STEP_CLASSES  # noqa: E402

# address "local:InitializeStep" -> the Step class (for docstrings).
# ALL_STEP_CLASSES is already a {class_name: class} dict.
_CLASS_BY_NAME = dict(ALL_STEP_CLASSES)


def _class_for(address: str):
    name = address.split(":")[-1] if address else ""
    return _CLASS_BY_NAME.get(name)


def build_spec() -> dict:
    doc = build_parca_document()
    spec: dict = {"name": "parca"}
    for name in STEP_ORDER:
        step = doc.get(name)
        if step is None:
            continue
        address = step.get("address", "")
        node = {
            "_type": "process",
            "address": address,
            "inputs": dict(step.get("inputs") or {}),
            "outputs": dict(step.get("outputs") or {}),
        }
        cls = _class_for(address)
        if cls is not None:
            # Standard formal-description hook: the `description` class attr,
            # falling back to the class docstring (steps aren't instantiated
            # here, so read at the class level rather than via describe()).
            d = getattr(cls, "description", "") or inspect.getdoc(cls) or ""
            if d:
                node["doc"] = d
        spec[name] = node
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/viz/parca_interactive.html")
    ap.add_argument("--max-row-width", type=int, default=1600)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    spec = build_spec()
    snippet = emit_html(spec, height="90vh", width="100%", inspector=True,
                        max_row_width=args.max_row_width)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>v2ecoli ParCa — pipeline structure</title>"
        "<style>body{margin:0;font-family:system-ui,sans-serif}"
        "h1{font:600 14px system-ui;margin:8px 12px;color:#334155}</style></head>"
        "<body><h1>v2ecoli ParCa pipeline — bigraph structure</h1>"
        f"{snippet}</body></html>"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.resolve()}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
