"""
v2ecoli — Mathematical Structure Report
=======================================

A standalone HTML report of the *mathematics* of the whole-cell model: every
process / deriver that carries a formal ``description`` class attribute, the
governing equations it states, grouped by biological subsystem, plus the
per-tick execution flow and the partition→allocate→evolve contract that wires
them together.

This is distinct from:
  - the interactive bigraph viewer (docs/bigraph_baseline.html) — topology/wiring;
  - the v1/v2 comparison report — engine benchmark / trajectory parity.

Builds purely from class attributes (no ParCa cache, no composite run), so it is
fast and never touches simulation state.

Usage:
    python reports/math_structure_report.py [--out docs/math_structure.html] [--open]
"""
from __future__ import annotations

import argparse
import html
import importlib
import inspect
import pkgutil
import re
import webbrowser
from pathlib import Path

import v2ecoli.processes as _processes_pkg
import v2ecoli.steps.derivers as _derivers_pkg


# ---------------------------------------------------------------------------
# Subsystem classification (mirrors v2ecoli/visualizations/_helpers.BIO_COLORS
# so the math report and the topology viewer use the same colour language).
# ---------------------------------------------------------------------------
SUBSYSTEMS = [
    ("replication", "DNA replication", "#F4A7A1",
        lambda n: any(k in n for k in ("chromosome", "replication", "oriC", "dnaa"))),
    ("transcription", "Transcription", "#9CC8E3",
        lambda n: "transcript" in n or "rnap" in n or "rna_synth" in n),
    ("rna", "RNA metabolism", "#B5D9F0",
        lambda n: n.startswith("rna") or "rna-" in n or "rna_deg" in n or "rna_matur" in n),
    ("translation", "Translation", "#A4D4A4",
        lambda n: "polypeptide" in n or "ribosome" in n or "translation" in n
                  or "protein" in n),
    ("regulation", "Regulation (TFs, ppGpp)", "#D9B8E0",
        lambda n: "tf_" in n or "tf-" in n or "ppgpp" in n or "attenuation" in n),
    ("signaling", "Signaling / equilibrium", "#FFD58C",
        lambda n: "equilibrium" in n or "two_component" in n or "two-component" in n
                  or "complexation" in n),
    ("metabolism", "Metabolism (FBA)", "#F7D488",
        lambda n: "metabolism" in n),
    ("counts", "Derived properties", "#D5D5D5",
        lambda n: "counts" in n or "mass" in n or "deriver" in n),
]
DEFAULT_GROUP = ("other", "Other", "#E8E8E8", lambda n: True)


def classify(class_name: str, step_name: str) -> tuple[str, str, str]:
    key = (class_name + " " + (step_name or "")).lower()
    for sub_key, label, color, matcher in SUBSYSTEMS:
        if matcher(key):
            return sub_key, label, color
    return DEFAULT_GROUP[0], DEFAULT_GROUP[1], DEFAULT_GROUP[2]


# ---------------------------------------------------------------------------
# Collect described classes
# ---------------------------------------------------------------------------
def collect_described():
    """Return [{class, step_name, module, description}] for every process /
    deriver class that defines a non-empty ``description`` string."""
    out = []
    seen = set()
    for pkg in (_processes_pkg, _derivers_pkg):
        for m in pkgutil.iter_modules(pkg.__path__):
            try:
                mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
            except Exception:
                continue
            for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                desc = getattr(cls, "description", None)
                if not isinstance(desc, str) or not desc.strip():
                    continue
                if cls_name in seen:
                    continue
                seen.add(cls_name)
                out.append({
                    "class": cls_name,
                    "step_name": getattr(cls, "name", "") or "",
                    "module": mod.__name__.split(".")[-1],
                    "description": inspect.cleandoc(desc),
                })
    return out


# ---------------------------------------------------------------------------
# Execution flow + partition contract (structural context for the math)
# ---------------------------------------------------------------------------
def execution_flow():
    from v2ecoli.composites.baseline import build_execution_layers, DEFAULT_FEATURES
    layers = build_execution_layers(DEFAULT_FEATURES)
    rows = []
    for L in layers:
        if not isinstance(L, list):
            continue
        steps = [s for s in L if not s.startswith("unique_update")]
        if steps:
            rows.append(steps)
    return rows


def partitioned_processes():
    from v2ecoli.composites._helpers import PARTITIONED_PROCESSES
    return sorted(PARTITIONED_PROCESSES.keys())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_MATH_LINE = re.compile(r"^\s{2,}\S")  # indented line ⇒ render as an equation block


def render_description(desc: str) -> str:
    """Turn a description string into HTML: indented lines become monospace
    equation blocks; blank lines split paragraphs; the rest is prose."""
    parts = []
    buf_math = []
    buf_prose = []

    def flush_prose():
        if buf_prose:
            parts.append("<p>" + html.escape(" ".join(buf_prose)) + "</p>")
            buf_prose.clear()

    def flush_math():
        if buf_math:
            body = "\n".join(html.escape(line) for line in buf_math)
            parts.append(f"<pre class='eq'>{body}</pre>")
            buf_math.clear()

    for line in desc.split("\n"):
        if not line.strip():
            flush_math(); flush_prose()
            continue
        if _MATH_LINE.match(line):
            flush_prose()
            buf_math.append(line.rstrip())
        else:
            flush_math()
            buf_prose.append(line.strip())
    flush_math(); flush_prose()
    return "\n".join(parts)


CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, system-ui, sans-serif; color: #1e293b;
       background: #f8fafc; line-height: 1.55; }
header { background: #0f172a; color: #f1f5f9; padding: 28px 40px; }
header h1 { margin: 0 0 6px; font-size: 24px; }
header p { margin: 0; color: #94a3b8; font-size: 14px; max-width: 70ch; }
main { max-width: 1000px; margin: 0 auto; padding: 32px 40px 80px; }
.toc { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 28px; }
.toc a { text-decoration: none; font-size: 12px; padding: 4px 10px; border-radius: 12px;
         color: #0f172a; border: 1px solid #cbd5e1; }
h2 { font-size: 18px; margin: 34px 0 4px; padding: 6px 12px; border-radius: 6px; }
h2 .count { font-size: 12px; color: #475569; font-weight: 400; }
.proc { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 16px 20px; margin: 14px 0; }
.proc h3 { margin: 0 0 2px; font-size: 16px; }
.proc .meta { font-size: 12px; color: #64748b; margin: 0 0 10px;
              font-family: ui-monospace, monospace; }
.proc p { margin: 8px 0; font-size: 14px; }
pre.eq { background: #0f172a; color: #e2e8f0; padding: 12px 14px; border-radius: 6px;
         overflow-x: auto; font-size: 13px; font-family: ui-monospace, "SF Mono", monospace;
         margin: 8px 0; }
section.flow { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
               padding: 16px 20px; margin: 14px 0; }
.layer { font-family: ui-monospace, monospace; font-size: 12px; padding: 2px 0;
         border-bottom: 1px dashed #e2e8f0; }
.layer .n { color: #94a3b8; display: inline-block; width: 34px; }
.tag { display: inline-block; background: #eef2ff; color: #3730a3; border-radius: 4px;
       padding: 1px 6px; margin: 1px 3px 1px 0; font-size: 11px; }
.note { font-size: 13px; color: #475569; }
"""


def build_html(described, flow, partitioned) -> str:
    # group described classes by subsystem
    groups: dict[str, dict] = {}
    for d in described:
        key, label, color = classify(d["class"], d["step_name"])
        g = groups.setdefault(key, {"label": label, "color": color, "items": []})
        g["items"].append(d)
    # stable subsystem order from SUBSYSTEMS, then any leftovers
    order = [s[0] for s in SUBSYSTEMS] + ["other"]
    ordered = [(k, groups[k]) for k in order if k in groups]

    toc = "".join(
        f"<a href='#{k}' style='background:{g['color']}33'>{html.escape(g['label'])}</a>"
        for k, g in ordered
    )

    sections = []
    for k, g in ordered:
        items = sorted(g["items"], key=lambda d: d["class"])
        cards = []
        for d in items:
            sname = f" · <code>{html.escape(d['step_name'])}</code>" if d["step_name"] else ""
            cards.append(
                f"<div class='proc'><h3>{html.escape(d['class'])}</h3>"
                f"<p class='meta'>{html.escape(d['module'])}.py{sname}</p>"
                f"{render_description(d['description'])}</div>"
            )
        sections.append(
            f"<h2 id='{k}' style='background:{g['color']}55'>{html.escape(g['label'])} "
            f"<span class='count'>· {len(items)} process"
            f"{'es' if len(items)!=1 else ''}</span></h2>" + "".join(cards)
        )

    # execution flow
    flow_rows = []
    for i, steps in enumerate(flow):
        tags = "".join(f"<span class='tag'>{html.escape(s)}</span>" for s in steps)
        flow_rows.append(f"<div class='layer'><span class='n'>{i}</span>{tags}</div>")
    part_tags = "".join(f"<span class='tag'>{html.escape(p)}</span>" for p in partitioned)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v2ecoli — Mathematical Structure</title><style>{CSS}</style></head>
<body>
<header>
  <h1>v2ecoli — Mathematical Structure</h1>
  <p>The governing equations of the whole-cell <i>E. coli</i> model, by subsystem.
     Each entry is the formal description declared on its process/deriver class.
     {len(described)} processes carry formal math.</p>
</header>
<main>
  <div class="toc">{toc}</div>
  {''.join(sections)}

  <h2 id="flow" style="background:#e2e8f0">Per-tick execution flow
    <span class="count">· {len(flow)} ordered layers</span></h2>
  <section class="flow">
    <p class="note">Steps in the same layer see the same starting state (per-layer
    atomicity); each layer's writes are reconciled before the next fires.
    <code>unique_update</code> flush barriers are omitted for legibility.</p>
    {''.join(flow_rows)}
  </section>

  <h2 id="partition" style="background:#FFAE8055">Partition → allocate → evolve
    <span class="count">· {len(partitioned)} partitioned processes</span></h2>
  <section class="flow">
    <p class="note">These processes compete for shared bulk molecules. Each is split
    into a <b>Requester</b> (computes demand from <code>bulk_total</code>), an
    <b>Allocator</b> (partitions the shared pool by priority), and an
    <b>Evolver</b> (acts on its <code>allocate</code>d share):</p>
    <p>{part_tags}</p>
  </section>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/math_structure.html")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    described = collect_described()
    flow = execution_flow()
    partitioned = partitioned_processes()

    page = build_html(described, flow, partitioned)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({len(described)} processes, {len(flow)} layers, "
          f"{len(partitioned)} partitioned)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
