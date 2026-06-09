#!/usr/bin/env python
"""Interactive bigraph-viz2 view of the v2ecoli baseline composite structure.

Builds the baseline composite, transforms its live state into a JSON-safe
*structural* spec (processes + wiring + store hierarchy, with huge data stores
collapsed), and renders a self-contained interactive HTML page via bigraph-viz2.

Run:  .venv/bin/python scripts/viz_baseline_interactive.py [--open]
Out:  out/viz/baseline_interactive.html
"""
from __future__ import annotations

import argparse
import inspect
import sys
import webbrowser
from pathlib import Path

# bigraph-viz2 is a sibling repo (pure Python, vendored JS bundle) — import by path.
import v2ecoli  # noqa: E402

try:
    from bigraph_viz2 import emit_html
except ModuleNotFoundError:  # fall back to a sibling checkout of bigraph-viz2
    _sib = Path(__file__).resolve().parents[2] / "bigraph-viz2" / "py"
    if _sib.is_dir():
        sys.path.insert(0, str(_sib))
    from bigraph_viz2 import emit_html  # noqa: E402

# Stores that hold thousands of molecules — show as a single collapsed node, not
# their contents, so the structural view stays legible.
COLLAPSE_STORES = {"bulk"}
# A store with more than this many children is summarized rather than expanded.
MAX_CHILDREN = 80
PROCESS_TYPES = {"process", "step"}


def _short_value(v):
    """A compact, JSON-safe preview of a leaf value (or None to omit)."""
    if isinstance(v, (int, float, bool, str)):
        s = str(v)
        return s if len(s) <= 40 else s[:37] + "..."
    return None


def _schema_view(schema):
    """Reduce a process-bigraph port schema to a JSON-safe tree of type strings.

    Leaves like ``{"_type": "quantity[float,fg]", "_default": 0.0}`` collapse to
    just ``"quantity[float,fg]"`` (the part that declares the type and units);
    nested sub-stores recurse. Bare type strings pass through.
    """
    if isinstance(schema, str):
        return schema
    if isinstance(schema, dict):
        if "_type" in schema:
            t = schema["_type"]
            return t if isinstance(t, str) else str(t)
        return {k: _schema_view(v) for k, v in schema.items()}
    return str(schema)


def _clean_wires(wires: dict) -> dict:
    """Drop synthetic flow-ordering tokens (_layer_*) from a process's ports."""
    return {
        port: path
        for port, path in wires.items()
        if not port.startswith("_layer_")
    }


def to_structural_spec(node, name=""):
    """Recursively convert a live-state dict node into a JSON-safe viz spec node."""
    if not isinstance(node, dict):
        # Leaf data value -> a variable node.
        out = {"_type": "variable"}
        val = _short_value(node)
        if val is not None:
            out["value"] = val
        return out

    ntype = node.get("_type")
    if ntype in PROCESS_TYPES:
        spec = {
            "_type": "process",
            "address": node.get("address", ""),
            "inputs": _clean_wires(node.get("inputs", {}) or {}),
            "outputs": _clean_wires(node.get("outputs", {}) or {}),
        }
        # Pull the declared port schemas (with units) off the process instance.
        # config in the live state is usually empty (params are baked into the
        # instance at construction), so the interface schema is what carries the
        # types/units worth showing.
        inst = node.get("instance")
        if inst is not None:
            try:
                if hasattr(inst, "inputs"):
                    in_schema = _schema_view(inst.inputs() or {})
                    if in_schema:
                        spec["inputSchema"] = in_schema
                if hasattr(inst, "outputs"):
                    out_schema = _schema_view(inst.outputs() or {})
                    if out_schema:
                        spec["outputSchema"] = out_schema
            except Exception:
                pass
            # Formal description via the standard Edge.describe() hook
            # (the `description` class attr, falling back to the docstring).
            try:
                if hasattr(inst, "describe"):
                    doc = inst.describe()
                else:
                    doc = inspect.getdoc(inst) or ""
                if doc:
                    spec["doc"] = doc
            except Exception:
                pass
        return spec

    # Otherwise: a store (container).
    if name in COLLAPSE_STORES:
        return {"_type": "store", f"({len(node)} molecules)": {"_type": "variable"}}

    children = {k: v for k, v in node.items() if not k.startswith("_")}
    if len(children) > MAX_CHILDREN:
        return {
            "_type": "store",
            f"({len(children)} entries — collapsed)": {"_type": "variable"},
        }

    store = {"_type": "store"}
    for k, v in children.items():
        store[k] = to_structural_spec(v, name=k)
    return store


def _estimate_row_width(spec: dict, collapsed: list[str]) -> int:
    """Pick a wrap width that makes the overall layout roughly square.

    The renderer wraps every store's children at `max_row_width`; at the default
    480 the whole model stacks into one tall ribbon. We estimate the visible
    node area (treating collapsed stores as chip-sized, since their children are
    hidden) and target a row width near its square root so rows pack side-by-side.
    """
    SKIP = {"_type", "name", "address", "inputs", "outputs",
            "inputSchema", "outputSchema"}
    hidden = set(collapsed)
    root = spec.get("name", "baseline")

    def area(node, node_id: str) -> float:
        if not isinstance(node, dict):
            return 0.0
        if node_id in hidden:
            return 140 * 56          # collapsed chip
        t = node.get("_type")
        if t == "variable":
            return 56 * 46
        if t == "process":
            return 260 * 52          # processes are wide (long addresses)
        a = 1500.0                   # store header/padding overhead
        for k, v in node.items():
            if k not in SKIP and isinstance(v, dict):
                a += area(v, f"{node_id}/{k}")
        return a

    total = area(spec, root)
    target = int((total ** 0.5) * 1.4)
    return max(800, min(3600, target))


def _top_store_ids(spec: dict) -> list[str]:
    """Ids of the root's direct sub-stores, so they start collapsed as chips.

    This makes the initial view show processes + store-chips + their wiring
    (the top-level structure) instead of hundreds of leaf variables; the user
    double-clicks a chip to expand it.
    """
    root = spec.get("name", "baseline")
    return [
        f"{root}/{k}"
        for k, v in spec.items()
        if k not in ("_type", "name")
        and isinstance(v, dict) and v.get("_type") == "store"
    ]


_META_KEYS = {"_type", "name", "address", "inputs", "outputs",
              "inputSchema", "outputSchema", "type", "value"}


def _annotate_types(spec: dict, root_name: str) -> None:
    """Assign each store/variable node a declared process-bigraph type.

    The only reliable source of types/units is the process port schemas. Each
    port's schema describes the type tree at the store it wires to, so we walk
    every process's (wire path, schema) pairs and overlay the leaf type strings
    onto the nodes at those paths.
    """
    type_by_path: dict[str, str] = {}

    def overlay(schema, path: str) -> None:
        if isinstance(schema, str):
            type_by_path.setdefault(path, schema)
        elif isinstance(schema, dict):
            for k, v in schema.items():
                overlay(v, f"{path}/{k}")

    def collect(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("_type") == "process":
            wires = {**(node.get("inputs") or {}), **(node.get("outputs") or {})}
            schemas = {**(node.get("inputSchema") or {}), **(node.get("outputSchema") or {})}
            for port, wire in wires.items():
                sch = schemas.get(port)
                if sch is not None and isinstance(wire, list):
                    overlay(sch, f"{root_name}/" + "/".join(str(s) for s in wire))
            return
        for k, v in node.items():
            if k not in _META_KEYS and isinstance(v, dict):
                collect(v, f"{path}/{k}")

    def assign(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        t = type_by_path.get(path)
        if t is not None and "type" not in node:
            node["type"] = t
        for k, v in node.items():
            if k not in _META_KEYS and isinstance(v, dict):
                assign(v, f"{path}/{k}")

    collect(spec, root_name)
    assign(spec, root_name)


def build_spec(seed: int, cache_dir: str) -> dict:
    composite = v2ecoli.build_composite("baseline", seed=seed, cache_dir=cache_dir)
    root = composite.state["agents"]["0"]
    spec = to_structural_spec(root, name="baseline")
    spec["name"] = "baseline"
    _annotate_types(spec, "baseline")
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--out", default="out/viz/baseline_interactive.html")
    ap.add_argument("--max-row-width", type=int, default=0,
                    help="row-wrap width in px; 0 = auto (square-ish layout)")
    ap.add_argument("--expand-all", action="store_true",
                    help="start fully expanded (default collapses top-level stores)")
    ap.add_argument("--open", action="store_true", help="open in browser when done")
    args = ap.parse_args()

    spec = build_spec(args.seed, args.cache_dir)
    collapsed = [] if args.expand_all else _top_store_ids(spec)
    mrw = args.max_row_width or _estimate_row_width(spec, collapsed)
    print(f"max_row_width = {mrw}; collapsed {len(collapsed)} top-level stores")

    snippet = emit_html(spec, height="90vh", width="100%", inspector=True,
                        max_row_width=mrw, collapsed=collapsed)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>v2ecoli baseline — bigraph structure</title>"
        "<style>body{margin:0;font-family:system-ui,sans-serif}"
        "h1{font:600 14px system-ui;margin:8px 12px;color:#334155}</style></head>"
        "<body><h1>v2ecoli baseline composite — bigraph structure</h1>"
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
