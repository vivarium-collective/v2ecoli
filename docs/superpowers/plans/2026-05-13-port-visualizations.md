# Port v2ecoli Reports to Visualization Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port each of v2ecoli's 7 report scripts to a registered `Visualization(Step)` subclass under `v2ecoli/visualizations/`, with rendering logic owned by the Step and `reports/*.py` reduced to thin orchestration wrappers.

**Architecture:** Create a new `v2ecoli/visualizations/` subpackage parallel to `v2ecoli/composites/`. Each `<name>.py` exposes a `<Name>Visualization(Visualization)` subclass with declared `inputs()` and `update(state) -> {'html': str}`. Shared scaffolding (Cytoscape graph rendering, repro banner, HTML document wrapper, trajectory utilities) lives in `v2ecoli/visualizations/_helpers.py`. Each `reports/<name>_report.py` becomes a ~40-line wrapper: build composite → run → extract state → dispatch to Step → write HTML.

**Tech Stack:** Python 3.12, process-bigraph, bigraph-schema, pbg-superpowers (Visualization base), pytest, BeautifulSoup (for HTML parity diffs).

**Branch:** `port-visualizations` (already cut from `main @ 5e30aa0`, spec committed as `7a0a949`).

**Spec:** `docs/superpowers/specs/2026-05-13-port-visualizations-design.md`.

---

## File Structure

Files created, modified, or deleted by this plan:

| File | Responsibility |
|---|---|
| `v2ecoli/visualizations/__init__.py` | Forces import of all 7 Step modules so discovery side-effects fire. New file. |
| `v2ecoli/visualizations/_helpers.py` | Shared infra: `classify`, `build_graph`, `render_cytoscape_html` (migrated from `v2ecoli/viz/network.py`); new `render_repro_banner`, `render_document`, `history_to_arrays`, `group_by_generation`, `group_by_agent`. New file. |
| `v2ecoli/visualizations/network.py` | `NetworkVisualization` — structural, renders one composite's Cytoscape diagram. New file. |
| `v2ecoli/visualizations/compare.py` | `CompareVisualization` — three side-by-side Cytoscape diagrams. New file. |
| `v2ecoli/visualizations/workflow.py` | `WorkflowVisualization` — full lifecycle report. New file. |
| `v2ecoli/visualizations/multigeneration.py` | `MultigenerationVisualization` — N-generation lineage. New file. |
| `v2ecoli/visualizations/colony.py` | `ColonyVisualization` — pymunk colony report. New file. |
| `v2ecoli/visualizations/benchmark.py` | `BenchmarkVisualization` — v2ecoli vs vEcoli composite. New file. |
| `v2ecoli/visualizations/v1_v2.py` | `V1V2Visualization` — three-way comparison. New file. |
| `reports/network_report.py` | Rewritten as ~40-line wrapper. |
| `reports/compare_report.py` | Rewritten as wrapper (3-architecture orchestration stays in script). |
| `reports/workflow_report.py` | Rewritten as wrapper. |
| `reports/multigeneration_report.py` | Rewritten as wrapper. |
| `reports/colony_report.py` | Rewritten as wrapper. |
| `reports/benchmark_report.py` | Rewritten as wrapper (vEcoli subprocess stays in script). |
| `reports/v1_v2_report.py` | Rewritten as wrapper. |
| `v2ecoli/viz/network.py` | DELETED. |
| `v2ecoli/viz/__init__.py` | DELETED. |
| `v2ecoli/viz/` | Empty directory removed. |
| `scripts/viz_baseline.py` | Imports updated to point at `v2ecoli.visualizations._helpers`. |
| `scripts/viz_network.py` | Imports updated to point at `v2ecoli.visualizations._helpers`. |
| `scripts/regenerate_viz_goldens.py` | New: runs each `reports/*.py` against current `main` to capture golden HTML. |
| `tests/fixtures/visualizations/*.golden.html` | 7 committed golden HTML files. |
| `tests/test_visualizations_network.py` | Unit test for `NetworkVisualization`. New file. |
| `tests/test_visualizations_compare.py` | Unit test for `CompareVisualization`. New file. |
| `tests/test_visualizations_workflow.py` | Unit test for `WorkflowVisualization`. New file. |
| `tests/test_visualizations_multigeneration.py` | Unit test. New file. |
| `tests/test_visualizations_colony.py` | Unit test. New file. |
| `tests/test_visualizations_benchmark.py` | Unit test. New file. |
| `tests/test_visualizations_v1_v2.py` | Unit test. New file. |
| `tests/test_visualizations_html_parity.py` | `@pytest.mark.sim` parity tests against goldens. New file. |
| `tests/test_visualizations_discovery.py` | `@pytest.mark.fast` smoke test that all 7 Steps appear in `core.link_registry`. New file. |
| `tests/test_visualizations_helpers.py` | Helper-level tests for `_helpers.py`. New file. |

---

## Task 1: Infrastructure setup — `_helpers.py`, `__init__.py`, capture goldens

This task lays the foundation. It MUST happen first so the per-Step ports in Tasks 2-8 have somewhere to land their imports.

**Files:**
- Create: `v2ecoli/visualizations/__init__.py`
- Create: `v2ecoli/visualizations/_helpers.py`
- Create: `scripts/regenerate_viz_goldens.py`
- Create: `tests/fixtures/visualizations/*.golden.html` (7 files, captured from current main)
- Create: `tests/test_visualizations_helpers.py`

### Step 1: Write the failing helper tests

Create `tests/test_visualizations_helpers.py`:

```python
"""Unit tests for v2ecoli.visualizations._helpers."""

import pytest


@pytest.mark.fast
def test_classify_returns_known_subsystem():
    from v2ecoli.visualizations._helpers import classify
    key, label = classify("ecoli-equilibrium")
    assert key in {"replication", "transcription", "rna", "translation",
                   "regulation", "signaling", "metabolism", "alloc",
                   "listen", "infra"}
    assert isinstance(label, str) and label


@pytest.mark.fast
def test_render_document_wraps_html():
    from v2ecoli.visualizations._helpers import render_document
    html = render_document(title="Test Page", body_html="<p>hello</p>",
                            include_banner=False)
    assert html.startswith("<!doctype html>") or html.startswith("<!DOCTYPE html>") \
        or html.startswith("<html")
    assert "<p>hello</p>" in html
    assert "Test Page" in html


@pytest.mark.fast
def test_render_repro_banner_includes_python_version():
    import sys
    from v2ecoli.visualizations._helpers import render_repro_banner
    banner = render_repro_banner()
    assert isinstance(banner, str)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert py_version in banner


@pytest.mark.fast
def test_history_to_arrays_extracts_paths():
    """history_to_arrays takes list-of-dict trajectory + list of dotted paths,
    returns {path: np.array} where each array is the column of values."""
    import numpy as np
    from v2ecoli.visualizations._helpers import history_to_arrays
    history = [
        {"a": 1.0, "b": {"c": 2.0}},
        {"a": 3.0, "b": {"c": 4.0}},
    ]
    arrs = history_to_arrays(history, ["a", "b.c"])
    assert "a" in arrs and "b.c" in arrs
    np.testing.assert_array_equal(arrs["a"], [1.0, 3.0])
    np.testing.assert_array_equal(arrs["b.c"], [2.0, 4.0])


@pytest.mark.fast
def test_group_by_generation_groups_rows():
    from v2ecoli.visualizations._helpers import group_by_generation
    history = [
        {"generation": 1, "v": 1},
        {"generation": 1, "v": 2},
        {"generation": 2, "v": 3},
    ]
    groups = group_by_generation(history)
    assert len(groups) == 2
    assert len(groups[0]) == 2  # gen 1 had 2 rows
    assert len(groups[1]) == 1  # gen 2 had 1 row
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.visualizations._helpers'`.

### Step 3: Create `v2ecoli/visualizations/__init__.py`

Create the file with:

```python
"""Visualization Steps for v2ecoli architectures.

Importing this package forces each per-Step module to load, which fires
their bigraph-schema link-registry side-effects via Step subclass
discovery. After ``import v2ecoli``, all Visualization Steps are
auto-registered in any ``allocate_core()``'s ``link_registry``.

Steps are added one at a time during the port; this file's imports
populate as each Step lands.
"""

# Step imports populate as Tasks 2-8 land each Visualization.
# Example (added during Task 2):
#   from v2ecoli.visualizations import network  # noqa: F401

__all__: list[str] = []
```

Tasks 2-8 each append their Step import + `__all__` entry.

### Step 4: Create `v2ecoli/visualizations/_helpers.py`

Migrate the contents of `v2ecoli/viz/network.py` (1066 LOC) into `_helpers.py`, then add the new shared helpers.

Concrete steps:
1. Copy the entire body of `v2ecoli/viz/network.py` into `v2ecoli/visualizations/_helpers.py`. Preserve all functions: `classify`, `build_graph`, `render_html`, `write_outputs`, plus any module-level constants (`BIO_COLORS`, etc.).
2. Rename `render_html` → `render_cytoscape_html` to disambiguate from the new `render_document` (avoids name collision).
3. Add three new shared helpers at the bottom:

```python
import datetime as _dt
import os as _os
import platform as _platform
import socket as _socket
import subprocess as _subprocess
import sys as _sys
from html import escape as _escape


def render_repro_banner() -> str:
    """Small HTML snippet with date, git commit, host, user, Python version,
    platform. Intended to be injected at the top of every generated report
    for traceability.

    Ported pattern from v2ecoli/library/repro_banner.py.
    """
    now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=-5)))  # Eastern
    try:
        git_sha = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    user = _os.environ.get("USER") or _os.environ.get("USERNAME") or "unknown"
    host = _socket.gethostname()
    py_version = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
    return (
        '<div class="repro-banner" style="font-size: 10px; color: #888; '
        'padding: 4px 8px; border-top: 1px solid #eee; margin-top: 12px;">'
        f'Generated {_escape(now.strftime("%Y-%m-%d %H:%M %Z"))} '
        f'· git {_escape(git_sha)} '
        f'· {_escape(user)}@{_escape(host)} '
        f'· python {_escape(py_version)} '
        f'· {_escape(_platform.system())}'
        '</div>'
    )


def render_document(
    title: str,
    body_html: str,
    head_extra: str = "",
    include_banner: bool = True,
) -> str:
    """Wrap a body fragment in a complete HTML document with repro banner."""
    banner = render_repro_banner() if include_banner else ""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f'  <title>{_escape(title)}</title>\n'
        '  <style>body { font-family: -apple-system, sans-serif; margin: 0; padding: 0; }</style>\n'
        f"  {head_extra}\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        f"{banner}\n"
        "</body>\n"
        "</html>\n"
    )


def history_to_arrays(history: list[dict], paths: list[str]) -> dict:
    """Extract columns from a trajectory list-of-dicts.

    ``paths`` are dot-separated (e.g. ``"listeners.mass.cell_mass"``).
    Returns ``{path: np.ndarray}``.
    """
    import numpy as np
    out: dict[str, list] = {p: [] for p in paths}
    for row in history:
        for path in paths:
            value = row
            for key in path.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            out[path].append(value)
    return {p: np.array(v) for p, v in out.items()}


def group_by_generation(history: list[dict]) -> list[list[dict]]:
    """Group rows by integer ``generation`` key.

    Returns a list of lists ordered by generation index (1, 2, 3, ...).
    Rows lacking a ``generation`` key go to generation 0.
    """
    by_gen: dict[int, list[dict]] = {}
    for row in history:
        gen = int(row.get("generation", 0))
        by_gen.setdefault(gen, []).append(row)
    return [by_gen[g] for g in sorted(by_gen)]


def group_by_agent(history: list[dict]) -> dict[str, list[dict]]:
    """Group rows by ``agent_id`` string key."""
    by_agent: dict[str, list[dict]] = {}
    for row in history:
        aid = str(row.get("agent_id", ""))
        by_agent.setdefault(aid, []).append(row)
    return by_agent
```

### Step 5: Run helper tests to verify they pass

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_helpers.py -v`
Expected: 5 PASS.

### Step 6: Create the goldens-regeneration script

Create `scripts/regenerate_viz_goldens.py`:

```python
"""Regenerate the HTML golden fixtures by running each report on current main.

Captures HTML output of every reports/<name>_report.py into
tests/fixtures/visualizations/<name>.golden.html. These goldens are the
"current main" reference that the post-port Visualization Step's HTML
output is compared against by tests/test_visualizations_html_parity.py.

USAGE: cd v2ecoli && uv run python scripts/regenerate_viz_goldens.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "visualizations"
OUT_DIR = REPO_ROOT / "out" / "reports"

REPORTS = {
    "network":         ("reports/network_report.py", []),
    "compare":         ("reports/compare_report.py", []),
    "workflow":        ("reports/workflow_report.py", []),
    "multigeneration": ("reports/multigeneration_report.py", []),
    "colony":          ("reports/colony_report.py", []),
    "benchmark":       ("reports/benchmark_report.py", []),
    "v1_v2":           ("reports/v1_v2_report.py", []),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Regenerate only these names (default: all)")
    args = parser.parse_args()
    targets = args.only or list(REPORTS)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for name in targets:
        if name not in REPORTS:
            print(f"skip: unknown report '{name}'", file=sys.stderr)
            continue
        script, extra_args = REPORTS[name]
        out_html = OUT_DIR / f"{name}.html"
        cmd = ["uv", "run", "python", script, "--out", str(out_html), *extra_args]
        print(f"[{name}] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as e:
            failures.append((name, str(e)))
            continue
        if not out_html.exists():
            failures.append((name, f"no output at {out_html}"))
            continue
        golden = FIXTURES_DIR / f"{name}.golden.html"
        shutil.copy(out_html, golden)
        print(f"[{name}] wrote {golden}")
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for n, err in failures:
            print(f"  {n}: {err}", file=sys.stderr)
        return 1
    print(f"\n{len(targets)} golden(s) regenerated under {FIXTURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 7: Capture current-main goldens

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python scripts/regenerate_viz_goldens.py`

This produces 7 HTML files under `tests/fixtures/visualizations/`. Each is the current-main rendering of its corresponding report. These are the parity baselines.

If a particular report fails (e.g., `benchmark_report.py` needs vEcoli subprocess setup), surface the failure to the controller — it may need to be regenerated by hand or skipped via the `--only` flag during this task.

Expected total time: 5-15 minutes (sim runtime dominates).

If ALL 7 fail, something is broken on `main` — STOP and report.

### Step 8: Run all tests to confirm no regression

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest -m "not sim and not slow" tests/ 2>&1 | tail -5`
Expected: same baseline as `main` plus 5 new passing tests in `test_visualizations_helpers.py`.

### Step 9: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/ scripts/regenerate_viz_goldens.py \
        tests/fixtures/visualizations/ tests/test_visualizations_helpers.py
git commit -m "$(cat <<'EOF'
feat(visualizations): scaffold the visualizations package + capture goldens

Creates v2ecoli/visualizations/{__init__,_helpers}.py with:
  - migrated Cytoscape graph rendering from v2ecoli/viz/network.py
    (build_graph, classify, render_cytoscape_html)
  - new shared HTML scaffolding (render_repro_banner, render_document)
  - trajectory utilities (history_to_arrays, group_by_generation,
    group_by_agent) used by the upcoming Visualization Steps.

scripts/regenerate_viz_goldens.py captures HTML output of each of the 7
reports/*.py scripts on current main into
tests/fixtures/visualizations/<name>.golden.html. These goldens are the
parity baselines that tests/test_visualizations_html_parity.py will diff
against after each Visualization Step's port lands.

v2ecoli/viz/network.py is NOT deleted yet — it stays alongside _helpers.py
through the per-Step ports (Tasks 2-8). Task 9 removes it after the
remaining 2 callers (scripts/viz_baseline.py + scripts/viz_network.py) are
migrated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Port `NetworkVisualization` (98 LOC report — smallest)

`NetworkVisualization` is the simplest port: structural-only (no trajectory), small wrapper. Use it as the reference shape for the other 6 Steps.

**Files:**
- Create: `v2ecoli/visualizations/network.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add import)
- Modify: `reports/network_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_network.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_network.py`:

```python
"""Unit tests for v2ecoli.visualizations.network.NetworkVisualization."""

import pytest


@pytest.mark.fast
def test_network_visualization_is_visualization_subclass():
    from v2ecoli.visualizations.network import NetworkVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(NetworkVisualization, Visualization)


@pytest.mark.fast
def test_network_visualization_inputs_has_composite_spec():
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "composite_spec" in inputs


@pytest.mark.fast
def test_network_visualization_outputs_html():
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(config={"title": "test"}, core=allocate_core())
    outputs = viz.outputs()
    assert outputs == {"html": "string"}


@pytest.mark.fast
def test_network_visualization_renders_synthetic_spec():
    """Construct with a minimal synthetic composite_spec; assert HTML output."""
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(
        config={"title": "Synthetic Network"},
        core=allocate_core(),
    )
    result = viz.update({
        "composite_spec": {
            "architecture": "baseline",
            "nodes": [{"id": "step1", "label": "Step 1"}],
            "edges": [],
            "layers": [["step1"]],
        },
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Synthetic Network" in html
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.visualizations.network'`.

### Step 3: Create `v2ecoli/visualizations/network.py`

```python
"""NetworkVisualization — interactive Cytoscape.js diagram of one v2ecoli
architecture's composition.

Migrated rendering from reports/network_report.py + v2ecoli/viz/network.py.
The Step takes a ``composite_spec`` describing the architecture (nodes,
edges, layers) and returns a complete HTML document.

The wrapper at reports/network_report.py is responsible for:
  - building the composite via v2ecoli.build_composite(name)
  - extracting nodes/edges/layers via _helpers.build_graph
  - passing the resulting spec to this Step's update().
"""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import (
    render_cytoscape_html,
    render_document,
)


class NetworkVisualization(Visualization):
    """Render one architecture's Cytoscape network diagram."""

    config_schema = {
        **Visualization.config_schema,
        "subtitle": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "composite_spec": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        spec = state.get("composite_spec") or {}
        title = self.config.get("title") or "v2ecoli network"
        subtitle = self.config.get("subtitle") or spec.get("architecture", "")

        # render_cytoscape_html expects {nodes, edges, layers, legend}.
        # If the spec already has those, pass it through; otherwise extract.
        body = render_cytoscape_html(spec, title=title, subtitle=subtitle)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}
```

### Step 4: Add the import to `__init__.py`

Edit `v2ecoli/visualizations/__init__.py` — add the import and `__all__` entry:

```python
from v2ecoli.visualizations import network  # noqa: F401

__all__: list[str] = ["network"]
```

### Step 5: Run the unit test to verify it passes

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_network.py -v`
Expected: 4 PASS.

If `test_network_visualization_renders_synthetic_spec` fails because `render_cytoscape_html` can't handle the minimal synthetic spec, adjust the test's synthetic spec to whatever minimum shape `render_cytoscape_html` accepts. Read `_helpers.py:render_cytoscape_html` to understand the expected fields.

### Step 6: Rewrite `reports/network_report.py` as a thin wrapper

Replace the entire body of `reports/network_report.py` with:

```python
"""Standalone composition-diagram viewer for a v2ecoli composite.

Loads one architecture, builds the execution layers, extracts the
composition graph, and renders an HTML page with the interactive Cytoscape
network. The rendering is owned by
``v2ecoli.visualizations.network.NetworkVisualization``; this script handles
CLI args + composite construction + writing the HTML.

Usage:
    python reports/network_report.py                          # baseline (default)
    python reports/network_report.py --model departitioned
    python reports/network_report.py --model reconciled
    python reports/network_report.py --model baseline --no-open
    python reports/network_report.py --out out/network_baseline.html
"""

import argparse
import os
import subprocess
import sys


MODELS = {
    "baseline":      "Baseline (partitioned)",
    "departitioned": "Departitioned",
    "reconciled":    "Reconciled",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS.keys(), default="baseline")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default: out/reports/network_<model>.html)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    from bigraph_schema import allocate_core
    from v2ecoli import build_composite
    from v2ecoli.composites.baseline import build_execution_layers, DEFAULT_FEATURES
    # (departitioned + reconciled also expose build_execution_layers; this is
    # the right one for `baseline`. The other branches import the matching
    # module's helpers below.)
    from v2ecoli.visualizations._helpers import build_graph
    from v2ecoli.visualizations.network import NetworkVisualization

    # Build composite + extract layers per architecture.
    if args.model == "departitioned":
        from v2ecoli.composites.departitioned import (
            build_execution_layers as _bel, DEFAULT_FEATURES as _df,
        )
        layers = _bel(_df)
    elif args.model == "reconciled":
        from v2ecoli.composites.reconciled import (
            build_execution_layers as _bel, DEFAULT_FEATURES as _df,
        )
        layers = _bel(_df)
    else:
        layers = build_execution_layers(DEFAULT_FEATURES)
    composite = build_composite(args.model, seed=0, cache_dir="out/cache")
    spec = build_graph(composite, layers)
    spec["architecture"] = args.model

    out_path = args.out or f"out/reports/network_{args.model}.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    viz = NetworkVisualization(
        config={
            "title": f"v2ecoli network — {MODELS[args.model]}",
            "subtitle": args.model,
        },
        core=allocate_core(),
    )
    result = viz.update({"composite_spec": spec})
    with open(out_path, "w") as f:
        f.write(result["html"])
    print(f"wrote {out_path}")
    if not args.no_open and sys.platform == "darwin":
        subprocess.Popen(["open", out_path])


if __name__ == "__main__":
    main()
```

### Step 7: Verify the wrapper runs

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python reports/network_report.py --model baseline --out /tmp/network_test.html --no-open 2>&1 | tail -5`
Expected: prints `wrote /tmp/network_test.html`. The file exists and is non-empty.

If the script crashes with a layer-import error or composite-build error, the migration in step 6 didn't preserve the right architecture-specific behavior. Inspect and fix.

### Step 8: Run unit tests + full fast suite

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_network.py tests/test_visualizations_helpers.py -v 2>&1 | tail -10`
Expected: 9 PASS (4 network + 5 helpers).

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest -m "not sim and not slow" tests/ 2>&1 | tail -5`
Expected: no regression.

### Step 9: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/network.py v2ecoli/visualizations/__init__.py \
        reports/network_report.py tests/test_visualizations_network.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port NetworkVisualization (structural)

NetworkVisualization renders one architecture's Cytoscape network diagram.
Inputs are a ``composite_spec`` (nodes + edges + layers); output is a
complete HTML document via _helpers.render_document.

reports/network_report.py becomes a ~50-line wrapper: parse args, build
composite via build_composite, extract spec via _helpers.build_graph,
dispatch to NetworkVisualization.update(), write HTML.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Port `BenchmarkVisualization` (133 LOC report)

**Files:**
- Create: `v2ecoli/visualizations/benchmark.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `benchmark` import)
- Modify: `reports/benchmark_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_benchmark.py`

### Step 1: Read the existing `reports/benchmark_report.py`

Use Read to inspect the file. Identify:
- The vEcoli subprocess invocation (stays in the wrapper).
- The rendering function(s) that produce HTML from the two trajectories. These move into the Step.

### Step 2: Write the failing unit test

Create `tests/test_visualizations_benchmark.py`:

```python
"""Unit tests for v2ecoli.visualizations.benchmark.BenchmarkVisualization."""

import pytest


@pytest.mark.fast
def test_benchmark_visualization_subclass():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(BenchmarkVisualization, Visualization)


@pytest.mark.fast
def test_benchmark_inputs_has_two_history_ports():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from bigraph_schema import allocate_core
    viz = BenchmarkVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history_v2ecoli" in inputs
    assert "history_vecoli" in inputs


@pytest.mark.fast
def test_benchmark_renders_synthetic_inputs():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from bigraph_schema import allocate_core
    viz = BenchmarkVisualization(
        config={"title": "Benchmark Test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v2ecoli": [{"time": 0.0, "mass": 1.0}, {"time": 1.0, "mass": 1.5}],
        "history_vecoli":  [{"time": 0.0, "mass": 1.0}, {"time": 1.0, "mass": 1.5}],
        "metadata": {"v2ecoli_version": "test", "vecoli_version": "test"},
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 3: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`.

### Step 4: Create `v2ecoli/visualizations/benchmark.py`

```python
"""BenchmarkVisualization — v2ecoli vs vEcoli composite benchmark comparison.

Migrated rendering from reports/benchmark_report.py. The Step takes two
trajectories (one from each engine) and renders a side-by-side comparison
HTML report. The wrapper at reports/benchmark_report.py handles the
subprocess invocations + trajectory collection.
"""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document


class BenchmarkVisualization(Visualization):
    """Render side-by-side performance + behavior comparison."""

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history_v2ecoli": "list[map[any]]",
            "history_vecoli":  "list[map[any]]",
            "metadata":        "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        h_v2 = state.get("history_v2ecoli") or []
        h_ve = state.get("history_vecoli") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli vs vEcoli benchmark"
        body = self._render_body(h_v2, h_ve, meta)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, h_v2: list, h_ve: list, meta: dict) -> str:
        # Migrate the body of the HTML-rendering function from the legacy
        # reports/benchmark_report.py here. The legacy report's render
        # function takes the two trajectories and produces side-by-side
        # plots + a metrics table. Move that body verbatim, replacing
        # references to ``v2_history`` / ``vecoli_history`` with ``h_v2`` /
        # ``h_ve`` (or whatever the legacy code used).
        raise NotImplementedError(
            "Migrate body from reports/benchmark_report.py's HTML render function."
        )
```

**Migration**: Open `reports/benchmark_report.py`, locate the function(s) that produce HTML (probably named something like `render_report`, `make_html`, or inlined into `main`). Copy that body into `_render_body()`, substituting argument names per the docstring.

### Step 5: Add to `__init__.py`

Edit `v2ecoli/visualizations/__init__.py`:

```python
from v2ecoli.visualizations import network, benchmark  # noqa: F401

__all__: list[str] = ["network", "benchmark"]
```

### Step 6: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_benchmark.py -v`
Expected: 3 PASS.

### Step 7: Rewrite `reports/benchmark_report.py`

Replace the entire body of `reports/benchmark_report.py` with a thin wrapper:

```python
"""Benchmark: v2ecoli vs vEcoli composite branch.

Runs each engine in a separate subprocess, collects the two trajectories,
then dispatches to v2ecoli.visualizations.benchmark.BenchmarkVisualization
to render the comparison HTML.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_v2ecoli(cache_dir: str, seed: int) -> list[dict]:
    """Run v2ecoli baseline composite in a subprocess; return its trajectory.

    Migrate this from the legacy benchmark_report.py's v2ecoli-subprocess
    code. The legacy script likely shells out to a small runner that pickles
    the result; reproduce that here.
    """
    raise NotImplementedError("Migrate from reports/benchmark_report.py")


def _run_vecoli() -> list[dict]:
    """Run vEcoli composite in a subprocess; return its trajectory.

    Migrate from the legacy benchmark_report.py's vecoli-subprocess code.
    """
    raise NotImplementedError("Migrate from reports/benchmark_report.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/reports/benchmark.html")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="out/cache")
    args = parser.parse_args()

    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization

    h_v2 = _run_v2ecoli(args.cache_dir, args.seed)
    h_ve = _run_vecoli()

    viz = BenchmarkVisualization(
        config={"title": "v2ecoli vs vEcoli benchmark"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v2ecoli": h_v2,
        "history_vecoli":  h_ve,
        "metadata": {"seed": args.seed, "cache_dir": args.cache_dir},
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(result["html"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

**Migration**: Move the subprocess-orchestration body from the legacy `reports/benchmark_report.py` into `_run_v2ecoli` and `_run_vecoli`. Everything that was rendering HTML stays out (it's in the Step now).

### Step 8: Smoke-test the wrapper

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/benchmark_report.py`
Expected: silent success.

(A full end-to-end run requires `out/cache/` and a working vEcoli install. Skip the run-test here; the parity test in Task 9 covers it.)

### Step 9: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/benchmark.py v2ecoli/visualizations/__init__.py \
        reports/benchmark_report.py tests/test_visualizations_benchmark.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port BenchmarkVisualization (multi-trajectory)

BenchmarkVisualization renders v2ecoli vs vEcoli comparison from two
trajectory inputs (history_v2ecoli + history_vecoli). The wrapper at
reports/benchmark_report.py keeps the subprocess orchestration (one
per engine) and dispatches to the Step for rendering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Port `V1V2Visualization` (411 LOC report)

Same shape as Task 3, but with three trajectory inputs (v1, v2, v2ecoli) instead of two. Subprocess orchestration of three engines stays in the wrapper.

**Files:**
- Create: `v2ecoli/visualizations/v1_v2.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `v1_v2` import)
- Modify: `reports/v1_v2_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_v1_v2.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_v1_v2.py`:

```python
"""Unit tests for v2ecoli.visualizations.v1_v2.V1V2Visualization."""

import pytest


@pytest.mark.fast
def test_v1_v2_visualization_subclass():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(V1V2Visualization, Visualization)


@pytest.mark.fast
def test_v1_v2_inputs_has_three_history_ports():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from bigraph_schema import allocate_core
    viz = V1V2Visualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history_v1" in inputs
    assert "history_v2" in inputs
    assert "history_v2ecoli" in inputs


@pytest.mark.fast
def test_v1_v2_renders_synthetic_inputs():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from bigraph_schema import allocate_core
    viz = V1V2Visualization(
        config={"title": "v1 vs v2 vs v2ecoli"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v1":      [{"time": 0.0, "mass": 1.0}],
        "history_v2":      [{"time": 0.0, "mass": 1.0}],
        "history_v2ecoli": [{"time": 0.0, "mass": 1.0}],
        "metadata":        {},
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 2: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_v1_v2.py -v`
Expected: FAIL with `ModuleNotFoundError`.

### Step 3: Create `v2ecoli/visualizations/v1_v2.py`

Use the same pattern as Task 3's `benchmark.py`, but with three `history_*` input ports and the legacy `reports/v1_v2_report.py`'s rendering body migrated into `_render_body(h_v1, h_v2, h_v2ecoli, meta)`.

```python
"""V1V2Visualization — three-way comparison: vEcoli 1.0 vs 2.0 vs v2ecoli."""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from v2ecoli.visualizations._helpers import render_document


class V1V2Visualization(Visualization):
    """Render three-way comparison from three trajectory inputs."""

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history_v1":      "list[map[any]]",
            "history_v2":      "list[map[any]]",
            "history_v2ecoli": "list[map[any]]",
            "metadata":        "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        h_v1 = state.get("history_v1") or []
        h_v2 = state.get("history_v2") or []
        h_v2e = state.get("history_v2ecoli") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v1 vs v2 vs v2ecoli"
        body = self._render_body(h_v1, h_v2, h_v2e, meta)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, h_v1, h_v2, h_v2e, meta) -> str:
        # Migrate from reports/v1_v2_report.py's HTML-rendering function.
        raise NotImplementedError("Migrate from reports/v1_v2_report.py")
```

Open `reports/v1_v2_report.py`, find its HTML-rendering function (it produces the comparison page), and copy that body into `_render_body`. Adapt argument names.

### Step 4: Add to `__init__.py`

```python
from v2ecoli.visualizations import network, benchmark, v1_v2  # noqa: F401

__all__: list[str] = ["network", "benchmark", "v1_v2"]
```

### Step 5: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_v1_v2.py -v`
Expected: 3 PASS.

### Step 6: Rewrite `reports/v1_v2_report.py`

Same wrapper pattern as `benchmark_report.py` but with three subprocess invocations. Replace the body with a wrapper that:
1. Parses args.
2. Runs three subprocesses (`_run_v1`, `_run_v2`, `_run_v2ecoli`); each returns a trajectory.
3. Instantiates `V1V2Visualization`, calls `update()` with the three trajectories.
4. Writes the resulting HTML to `--out`.

Reuse the helper structure from Task 3. Migrate the subprocess-orchestration body from the legacy `reports/v1_v2_report.py`.

### Step 7: Smoke-test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/v1_v2_report.py`
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/v1_v2.py v2ecoli/visualizations/__init__.py \
        reports/v1_v2_report.py tests/test_visualizations_v1_v2.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port V1V2Visualization (3-trajectory comparison)

V1V2Visualization renders vEcoli 1.0 vs 2.0 vs v2ecoli from three trajectory
inputs. The wrapper at reports/v1_v2_report.py keeps the three-subprocess
orchestration and dispatches to the Step for rendering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Port `MultigenerationVisualization` (521 LOC report)

Single-trajectory; multi-generation grouping handled inside the Step via `_helpers.group_by_generation`.

**Files:**
- Create: `v2ecoli/visualizations/multigeneration.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `multigeneration`)
- Modify: `reports/multigeneration_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_multigeneration.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_multigeneration.py`:

```python
"""Unit tests for v2ecoli.visualizations.multigeneration.MultigenerationVisualization."""

import pytest


@pytest.mark.fast
def test_multigeneration_subclass():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(MultigenerationVisualization, Visualization)


@pytest.mark.fast
def test_multigeneration_inputs():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from bigraph_schema import allocate_core
    viz = MultigenerationVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_multigeneration_renders_synthetic_two_generations():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from bigraph_schema import allocate_core
    viz = MultigenerationVisualization(
        config={"title": "Multi-gen test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"generation": 1, "time": 0.0, "mass": 1.0},
            {"generation": 1, "time": 100.0, "mass": 2.0},
            {"generation": 2, "time": 100.0, "mass": 1.0},
            {"generation": 2, "time": 200.0, "mass": 2.0},
        ],
        "metadata": {"n_generations": 2},
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 2: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_multigeneration.py -v`
Expected: FAIL with `ModuleNotFoundError`.

### Step 3: Create `v2ecoli/visualizations/multigeneration.py`

```python
"""MultigenerationVisualization — N-generation lineage with mass trajectories."""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from v2ecoli.visualizations._helpers import (
    render_document,
    group_by_generation,
    history_to_arrays,
)


class MultigenerationVisualization(Visualization):
    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[any]]",
            "metadata": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli multigeneration"

        gens = group_by_generation(history)
        body = self._render_body(gens, meta)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, gens: list[list[dict]], meta: dict) -> str:
        # Migrate the HTML-rendering body from
        # reports/multigeneration_report.py. The legacy report produces
        # end-to-end mass-trajectory plots across generations + a
        # fold-change summary. ``gens`` is the list-of-lists grouping
        # already produced via group_by_generation.
        raise NotImplementedError("Migrate from reports/multigeneration_report.py")
```

Open `reports/multigeneration_report.py`, find the function(s) that produce HTML (likely named something like `render_html`, `make_report`, or inlined). Copy that body into `_render_body`, substituting whatever the legacy code calls the grouped trajectory for `gens`.

### Step 4: Add to `__init__.py`

```python
from v2ecoli.visualizations import network, benchmark, v1_v2, multigeneration  # noqa: F401

__all__: list[str] = ["network", "benchmark", "v1_v2", "multigeneration"]
```

### Step 5: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_multigeneration.py -v`
Expected: 3 PASS.

### Step 6: Rewrite `reports/multigeneration_report.py`

Same wrapper pattern: build composite via `build_composite("baseline", ...)`, run for N generations (the legacy script's multi-cycle loop), collect trajectory, dispatch to `MultigenerationVisualization`. Migrate the multi-generation run loop from the legacy script (it knows how to chain cell divisions).

```python
"""v2ecoli multigeneration report.

Runs a single cell to division, keeps one daughter, runs that to division,
... — for ``--generations`` cycles. Dispatches the resulting trajectory to
v2ecoli.visualizations.multigeneration.MultigenerationVisualization.
"""

import argparse
import os

from process_bigraph.emitter import gather_emitter_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--out", default="out/reports/multigeneration.html")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="out/cache")
    args = parser.parse_args()

    from bigraph_schema import allocate_core
    from v2ecoli import build_composite
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization

    # Migrate the multi-generation run loop from the legacy script: build
    # one composite, run to division, take one daughter as the next
    # composite's initial state, run, ... ``args.generations`` cycles.
    # Tag each row with its ``generation`` index. Concatenate into one
    # ``history`` list.

    history: list[dict] = []
    # ... migrated loop here ...

    viz = MultigenerationVisualization(
        config={"title": f"v2ecoli {args.generations}-generation"},
        core=allocate_core(),
    )
    result = viz.update({"history": history,
                          "metadata": {"n_generations": args.generations}})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(result["html"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

Open `reports/multigeneration_report.py` (current main) and migrate the multi-generation simulation loop into the placeholder. That loop is the script's reason to exist; preserve it.

### Step 7: Smoke-test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/multigeneration_report.py`
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/multigeneration.py v2ecoli/visualizations/__init__.py \
        reports/multigeneration_report.py tests/test_visualizations_multigeneration.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port MultigenerationVisualization

MultigenerationVisualization renders N-generation lineage from a single
trajectory grouped via _helpers.group_by_generation. reports/multigeneration_report.py
keeps the multi-cycle simulation loop and dispatches to the Step for rendering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Port `ColonyVisualization` (911 LOC report)

Multi-cell colony with pymunk physics. Trajectory has per-agent slicing handled internally via `_helpers.group_by_agent`.

**Files:**
- Create: `v2ecoli/visualizations/colony.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `colony`)
- Modify: `reports/colony_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_colony.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_colony.py`:

```python
"""Unit tests for v2ecoli.visualizations.colony.ColonyVisualization."""

import pytest


@pytest.mark.fast
def test_colony_subclass():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(ColonyVisualization, Visualization)


@pytest.mark.fast
def test_colony_inputs():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from bigraph_schema import allocate_core
    viz = ColonyVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_colony_renders_synthetic_colony_state():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from bigraph_schema import allocate_core
    viz = ColonyVisualization(
        config={"title": "Colony test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"time": 0.0, "agent_id": "0",  "x": 0.0, "y": 0.0, "length": 1.0},
            {"time": 0.0, "agent_id": "01", "x": 1.0, "y": 0.0, "length": 1.0},
        ],
        "metadata": {"colony_size": 2},
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 2: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_colony.py -v`
Expected: FAIL.

### Step 3: Create `v2ecoli/visualizations/colony.py`

```python
"""ColonyVisualization — multi-cell pymunk colony report."""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from v2ecoli.visualizations._helpers import (
    render_document,
    group_by_agent,
)


class ColonyVisualization(Visualization):
    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[any]]",
            "metadata": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli colony"

        by_agent = group_by_agent(history)
        body = self._render_body(history, by_agent, meta)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, history: list[dict], by_agent: dict, meta: dict) -> str:
        # Migrate the body from reports/colony_report.py's HTML-rendering
        # function. The legacy report produces colony snapshots, mass
        # trajectories per cell, and phylogeny coloring. Move that body
        # here, substituting by_agent / history / meta for whatever names
        # the legacy code used.
        raise NotImplementedError("Migrate from reports/colony_report.py")
```

### Step 4: Add to `__init__.py`

```python
from v2ecoli.visualizations import (
    network, benchmark, v1_v2, multigeneration, colony
)  # noqa: F401

__all__: list[str] = [
    "network", "benchmark", "v1_v2", "multigeneration", "colony",
]
```

### Step 5: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_colony.py -v`
Expected: 3 PASS.

### Step 6: Rewrite `reports/colony_report.py`

Same wrapper pattern. The legacy script orchestrates a multi-cell colony simulation (EcoliWCM + pymunk physics + surrogate cells). Preserve the simulation orchestration in the wrapper; move only the HTML rendering into the Step.

Concretely:
1. Read `reports/colony_report.py` (911 LOC).
2. Identify the simulation block (composite construction + `composite.run(...)` + colony state collection). This stays in the wrapper.
3. Identify the HTML-rendering function(s). These move into `ColonyVisualization._render_body`.
4. The wrapper assembles `history` from the colony trajectory (rows tagged by `agent_id`, `time`, position, mass, etc.) and dispatches.

### Step 7: Smoke-test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/colony_report.py`
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/colony.py v2ecoli/visualizations/__init__.py \
        reports/colony_report.py tests/test_visualizations_colony.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port ColonyVisualization

ColonyVisualization renders multi-cell colony state grouped by agent.
reports/colony_report.py keeps the EcoliWCM + pymunk + surrogate-cell
simulation orchestration and dispatches to the Step for rendering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Port `CompareVisualization` (1002 LOC report)

3-architecture side-by-side Cytoscape diagrams. Structural (no trajectory in v0).

**Files:**
- Create: `v2ecoli/visualizations/compare.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `compare`)
- Modify: `reports/compare_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_compare.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_compare.py`:

```python
"""Unit tests for v2ecoli.visualizations.compare.CompareVisualization."""

import pytest


@pytest.mark.fast
def test_compare_subclass():
    from v2ecoli.visualizations.compare import CompareVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(CompareVisualization, Visualization)


@pytest.mark.fast
def test_compare_inputs_has_three_composite_specs():
    from v2ecoli.visualizations.compare import CompareVisualization
    from bigraph_schema import allocate_core
    viz = CompareVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "composite_specs" in inputs


@pytest.mark.fast
def test_compare_renders_three_synthetic_specs():
    from v2ecoli.visualizations.compare import CompareVisualization
    from bigraph_schema import allocate_core
    viz = CompareVisualization(
        config={"title": "Architecture compare"},
        core=allocate_core(),
    )
    result = viz.update({
        "composite_specs": [
            {"architecture": "baseline",      "nodes": [], "edges": [], "layers": []},
            {"architecture": "departitioned", "nodes": [], "edges": [], "layers": []},
            {"architecture": "reconciled",    "nodes": [], "edges": [], "layers": []},
        ],
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 2: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_compare.py -v`
Expected: FAIL.

### Step 3: Create `v2ecoli/visualizations/compare.py`

```python
"""CompareVisualization — side-by-side Cytoscape diagrams for the three
v2ecoli architectures.
"""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from v2ecoli.visualizations._helpers import (
    render_cytoscape_html,
    render_document,
)


class CompareVisualization(Visualization):
    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "composite_specs": "list[map[any]]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        specs = state.get("composite_specs") or []
        title = self.config.get("title") or "v2ecoli architecture compare"
        body = self._render_body(specs)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, specs: list[dict]) -> str:
        # Migrate from reports/compare_report.py's three-way HTML-rendering
        # function. Each spec gets render_cytoscape_html(); the three are
        # laid out side-by-side in a flex/grid container.
        raise NotImplementedError("Migrate from reports/compare_report.py")
```

### Step 4: Add to `__init__.py`

```python
from v2ecoli.visualizations import (
    network, benchmark, v1_v2, multigeneration, colony, compare,
)  # noqa: F401

__all__: list[str] = [
    "network", "benchmark", "v1_v2", "multigeneration", "colony", "compare",
]
```

### Step 5: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_compare.py -v`
Expected: 3 PASS.

### Step 6: Rewrite `reports/compare_report.py`

The legacy script builds three composites (baseline + departitioned + reconciled) — possibly via multiprocessing for parallel build. Preserve that orchestration in the wrapper; pass the three resulting specs to `CompareVisualization`. The legacy HTML rendering with side-by-side Cytoscape diagrams moves to the Step.

### Step 7: Smoke-test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/compare_report.py`
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/compare.py v2ecoli/visualizations/__init__.py \
        reports/compare_report.py tests/test_visualizations_compare.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port CompareVisualization (structural 3-arch)

CompareVisualization renders three Cytoscape diagrams (baseline, departitioned,
reconciled) side-by-side from three composite specs. reports/compare_report.py
keeps the three-composite parallel-build orchestration and dispatches to the
Step for rendering. Time-series comparison is a follow-up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Port `WorkflowVisualization` (2452 LOC report — biggest)

Full cell lifecycle (42-min sim) with multi-step orchestration (parca → cache → composite → run → division). Substantial rendering logic. Apply the established pattern.

**Files:**
- Create: `v2ecoli/visualizations/workflow.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add `workflow`)
- Modify: `reports/workflow_report.py` (rewrite as wrapper)
- Create: `tests/test_visualizations_workflow.py`

### Step 1: Write the failing unit test

Create `tests/test_visualizations_workflow.py`:

```python
"""Unit tests for v2ecoli.visualizations.workflow.WorkflowVisualization."""

import pytest


@pytest.mark.fast
def test_workflow_subclass():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(WorkflowVisualization, Visualization)


@pytest.mark.fast
def test_workflow_inputs():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from bigraph_schema import allocate_core
    viz = WorkflowVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_workflow_renders_synthetic_trajectory():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from bigraph_schema import allocate_core
    viz = WorkflowVisualization(
        config={"title": "Workflow test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"time": 0.0,    "mass": 380.0},
            {"time": 1000.0, "mass": 450.0},
            {"time": 2500.0, "mass": 702.0},
        ],
        "metadata": {"duration": 2520, "seed": 0},
    })
    assert "html" in result
    assert "<html" in result["html"]
```

### Step 2: Run to verify failure

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_workflow.py -v`
Expected: FAIL.

### Step 3: Create `v2ecoli/visualizations/workflow.py`

```python
"""WorkflowVisualization — full 42-min cell lifecycle report."""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization
from v2ecoli.visualizations._helpers import (
    render_document,
    history_to_arrays,
)


class WorkflowVisualization(Visualization):
    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[any]]",
            "metadata": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli workflow"
        body = self._render_body(history, meta)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, history: list[dict], meta: dict) -> str:
        # Migrate from reports/workflow_report.py — the LARGEST report
        # rendering body in v2ecoli (mass trajectory, growth rate, division
        # markers, listener panels, etc.). Move the rendering function(s)
        # here. Helper extraction via history_to_arrays + custom plotting
        # logic per panel.
        raise NotImplementedError("Migrate from reports/workflow_report.py")
```

### Step 4: Add to `__init__.py`

Final `__init__.py` (after Task 8):

```python
"""Visualization Steps for v2ecoli architectures.

Importing this package forces each per-Step module to load, which fires
their bigraph-schema link-registry side-effects via Step subclass
discovery. After ``import v2ecoli``, all Visualization Steps are
auto-registered in any ``allocate_core()``'s ``link_registry``.
"""

from v2ecoli.visualizations import (
    network, benchmark, v1_v2, multigeneration, colony, compare, workflow,
)  # noqa: F401

__all__: list[str] = [
    "network", "benchmark", "v1_v2", "multigeneration", "colony", "compare", "workflow",
]
```

### Step 5: Run unit tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_workflow.py -v`
Expected: 3 PASS.

### Step 6: Rewrite `reports/workflow_report.py`

This is the biggest legacy script. The wrapper retains the step-based pipeline orchestration (parca → cache → composite → run → division) — that's the script's reason to exist. The body that produces HTML moves to the Step. Apply patience: read the whole 2452-line legacy script first, identify which functions are orchestration vs rendering, then move only rendering.

### Step 7: Smoke-test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run python -m py_compile reports/workflow_report.py`
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add v2ecoli/visualizations/workflow.py v2ecoli/visualizations/__init__.py \
        reports/workflow_report.py tests/test_visualizations_workflow.py
git commit -m "$(cat <<'EOF'
feat(visualizations): port WorkflowVisualization (full lifecycle)

WorkflowVisualization renders the full 42-min cell lifecycle report
(mass trajectory, growth rate, division markers, listener panels) from
a single trajectory. reports/workflow_report.py keeps the step-based
pipeline orchestration (parca → cache → composite → run → division) and
dispatches to the Step for rendering.

This completes the per-Step ports (Tasks 2-8). All 7 Visualization Steps
are now registered.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Delete `v2ecoli/viz/`, update remaining callers

By this point, no caller imports from `v2ecoli/viz/`. Verify, update the 2 known callers (`scripts/viz_baseline.py`, `scripts/viz_network.py`), delete the legacy file.

**Files:**
- Delete: `v2ecoli/viz/network.py`
- Delete: `v2ecoli/viz/__init__.py`
- Delete: `v2ecoli/viz/` (directory)
- Modify: `scripts/viz_baseline.py`
- Modify: `scripts/viz_network.py`

### Step 1: Find all callers of `v2ecoli.viz`

Run:
```bash
cd /Users/eranagmon/code/v2ecoli && rg "from v2ecoli\.viz\b|import v2ecoli\.viz\b" --type py
```
Expected: 2 hits — `scripts/viz_baseline.py` and `scripts/viz_network.py`. If MORE callers surface, fix them too.

### Step 2: Update `scripts/viz_baseline.py`

Open `scripts/viz_baseline.py`, find the `from v2ecoli.viz...` import lines, and replace with the matching imports from `v2ecoli.visualizations._helpers`. The function names migrated unchanged except `render_html` → `render_cytoscape_html`.

```python
# Before
from v2ecoli.viz import build_graph, render_html, write_outputs
# After
from v2ecoli.visualizations._helpers import (
    build_graph,
    render_cytoscape_html as render_html,   # local alias to keep call sites working
    write_outputs,
)
```

(Or update the call sites if `render_html` is only used once or twice; either way works.)

### Step 3: Update `scripts/viz_network.py`

Same pattern.

### Step 4: Verify zero `v2ecoli.viz` imports remain

Run:
```bash
cd /Users/eranagmon/code/v2ecoli && rg "from v2ecoli\.viz\b|import v2ecoli\.viz\b" --type py
```
Expected: zero hits.

### Step 5: Delete the legacy files + directory

```bash
cd /Users/eranagmon/code/v2ecoli && rm v2ecoli/viz/network.py v2ecoli/viz/__init__.py && rmdir v2ecoli/viz
```

### Step 6: Run the fast test suite

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest -m "not sim and not slow" tests/ 2>&1 | tail -5`
Expected: same baseline as before plus 21+ new Visualization tests passing.

### Step 7: Smoke-test the migrated scripts

```bash
cd /Users/eranagmon/code/v2ecoli && \
  uv run python -m py_compile scripts/viz_baseline.py scripts/viz_network.py
```
Expected: silent success.

### Step 8: Commit

```bash
cd /Users/eranagmon/code/v2ecoli
git add -A
git commit -m "$(cat <<'EOF'
refactor: delete v2ecoli/viz/, migrate remaining callers to visualizations/_helpers

After Tasks 2-8 ported every report's rendering to a Visualization Step,
the legacy v2ecoli/viz/network.py is no longer reachable from any
Step or report. The two remaining callers (scripts/viz_baseline.py and
scripts/viz_network.py) are updated to import the same helpers from
v2ecoli.visualizations._helpers.

This completes the atomic refactor — there is no half-state where both
viz APIs are usable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Discovery test + final acceptance sweep

**Files:**
- Create: `tests/test_visualizations_discovery.py`
- No code changes outside the test.

### Step 1: Write the discovery test

Create `tests/test_visualizations_discovery.py`:

```python
"""Smoke test: all 7 Visualization Steps register into core.link_registry."""

import pytest


EXPECTED_VISUALIZATION_CLASSES = {
    "NetworkVisualization",
    "CompareVisualization",
    "WorkflowVisualization",
    "MultigenerationVisualization",
    "ColonyVisualization",
    "BenchmarkVisualization",
    "V1V2Visualization",
}


@pytest.mark.fast
def test_all_visualizations_discoverable():
    import v2ecoli  # noqa: F401 — forces discovery side-effects
    from bigraph_schema import allocate_core
    core = allocate_core()
    found = {
        k.rsplit(".", 1)[-1]
        for k in core.link_registry
        if k.startswith("v2ecoli.visualizations.")
    }
    missing = EXPECTED_VISUALIZATION_CLASSES - found
    assert not missing, f"missing visualizations in link_registry: {missing}"


@pytest.mark.fast
def test_visualizations_are_visualization_subclasses():
    """Each registered Visualization class can be imported and is a
    pbg_superpowers.visualization.Visualization subclass."""
    import v2ecoli  # noqa
    from bigraph_schema import allocate_core
    from pbg_superpowers.visualization import Visualization
    core = allocate_core()
    for k, cls in core.link_registry.items():
        if not k.startswith("v2ecoli.visualizations."):
            continue
        assert issubclass(cls, Visualization), f"{k} is not a Visualization subclass"
```

### Step 2: Run the discovery test

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_discovery.py -v`
Expected: 2 PASS.

### Step 3: Run the full fast suite

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest -m "not sim and not slow" tests/ 2>&1 | tail -5`
Expected: all fast tests pass (existing baseline + ~25 new tests across `test_visualizations_*.py`).

### Step 4: Run the HTML parity tests

Create `tests/test_visualizations_html_parity.py`:

```python
"""@pytest.mark.sim parity tests: each Visualization Step's HTML output
structurally matches the golden captured from current main.

Goldens live at tests/fixtures/visualizations/<name>.golden.html and are
regenerated by scripts/regenerate_viz_goldens.py.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest


pytest.importorskip("bs4")


FIXTURES = Path(__file__).parent / "fixtures" / "visualizations"


def _structural_diff(html_a: str, html_b: str) -> list[str]:
    """Compare two HTML strings structurally. Returns a list of human-readable
    diff lines; empty list means equivalent.

    Ignores: timestamps, git SHAs, hostnames, dates (the repro banner is
    expected to differ run-to-run).
    """
    from bs4 import BeautifulSoup
    issues: list[str] = []
    a = BeautifulSoup(html_a, "html.parser")
    b = BeautifulSoup(html_b, "html.parser")

    # Strip the repro banner before comparing — it's expected to differ.
    for soup in (a, b):
        for el in soup.select(".repro-banner"):
            el.extract()

    a_tags = [t.name for t in a.find_all()]
    b_tags = [t.name for t in b.find_all()]
    if len(a_tags) != len(b_tags):
        issues.append(f"tag count: golden={len(a_tags)} fresh={len(b_tags)}")
    a_ids = {t.get("id") for t in a.find_all() if t.get("id")}
    b_ids = {t.get("id") for t in b.find_all() if t.get("id")}
    if a_ids != b_ids:
        issues.append(f"id set differs: only golden={a_ids - b_ids} only fresh={b_ids - a_ids}")
    a_classes = set()
    b_classes = set()
    for t in a.find_all():
        a_classes.update(t.get("class", []))
    for t in b.find_all():
        b_classes.update(t.get("class", []))
    if a_classes != b_classes:
        issues.append(f"class set differs: only golden={a_classes - b_classes} only fresh={b_classes - a_classes}")
    return issues


@pytest.mark.sim
@pytest.mark.parametrize("name", [
    "network",
    "compare",
    "workflow",
    "multigeneration",
    "colony",
    "benchmark",
    "v1_v2",
])
def test_html_parity(name: str, tmp_path):
    golden_path = FIXTURES / f"{name}.golden.html"
    if not golden_path.exists():
        pytest.skip(f"golden fixture missing: {golden_path}; run scripts/regenerate_viz_goldens.py")

    fresh_path = tmp_path / f"{name}.html"
    script_path = Path(__file__).parent.parent / "reports" / f"{name}_report.py"

    subprocess.run(
        ["uv", "run", "python", str(script_path), "--out", str(fresh_path)],
        check=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert fresh_path.exists(), f"report did not produce {fresh_path}"

    golden_html = golden_path.read_text()
    fresh_html = fresh_path.read_text()
    issues = _structural_diff(golden_html, fresh_html)
    assert not issues, f"HTML parity issues for {name}:\n  " + "\n  ".join(issues)
```

### Step 5: Run the parity tests

Run: `cd /Users/eranagmon/code/v2ecoli && uv run pytest tests/test_visualizations_html_parity.py -v -m sim 2>&1 | tail -25`
Expected: 7 PASS.

If any test fails:
- The structural diff output tells you what differs (tag count, IDs, classes).
- The most likely cause: the ported Step's `_render_body` is missing some of the legacy report's rendering logic (the migration in Task N was incomplete).
- Fix in the corresponding `v2ecoli/visualizations/<name>.py` and re-run.

This is the equivalence gate. ALL 7 must pass before the PR ships.

### Step 6: Run every acceptance-criterion probe

Run from `/Users/eranagmon/code/v2ecoli`:

```bash
# 1. Fast suite
uv run pytest -m "not sim and not slow" tests/ 2>&1 | tail -3

# 2. Parity tests
uv run pytest -m sim tests/test_visualizations_html_parity.py 2>&1 | tail -3

# 3. No legacy viz imports
rg "v2ecoli\.viz\.network|from v2ecoli\.viz\b" v2ecoli/ tests/ scripts/ reports/ --type py && \
  echo "FAIL: legacy viz imports" || echo "ok: no legacy viz imports"

# 4. Legacy files deleted
[ ! -e v2ecoli/viz/network.py ] && echo "ok: v2ecoli/viz/network.py deleted" || echo "FAIL"
[ ! -d v2ecoli/viz ] && echo "ok: v2ecoli/viz/ removed" || echo "FAIL"

# 5. All 7 Steps importable
uv run python -c "
from v2ecoli.visualizations.network import NetworkVisualization
from v2ecoli.visualizations.compare import CompareVisualization
from v2ecoli.visualizations.workflow import WorkflowVisualization
from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
from v2ecoli.visualizations.colony import ColonyVisualization
from v2ecoli.visualizations.benchmark import BenchmarkVisualization
from v2ecoli.visualizations.v1_v2 import V1V2Visualization
print('all 7 importable')
"

# 6. All 7 register in core.link_registry
uv run python -c "
import v2ecoli
from bigraph_schema import allocate_core
core = allocate_core()
expected = {'NetworkVisualization', 'CompareVisualization', 'WorkflowVisualization',
            'MultigenerationVisualization', 'ColonyVisualization',
            'BenchmarkVisualization', 'V1V2Visualization'}
found = {k.rsplit('.',1)[-1] for k in core.link_registry if k.startswith('v2ecoli.visualizations.')}
print('ok' if expected <= found else f'FAIL: missing {expected - found}')
"

# 7. All wrappers py_compile
uv run python -m py_compile reports/network_report.py reports/compare_report.py \
    reports/workflow_report.py reports/multigeneration_report.py \
    reports/colony_report.py reports/benchmark_report.py reports/v1_v2_report.py

# 8. Spec + plan committed
[ -f docs/superpowers/specs/2026-05-13-port-visualizations-design.md ] && echo "ok: spec"
[ -f docs/superpowers/plans/2026-05-13-port-visualizations.md ] && echo "ok: plan"
```

Each line should print `ok: ...` or PASS-equivalent.

### Step 7: Commit (test file + acceptance summary)

```bash
cd /Users/eranagmon/code/v2ecoli
git add tests/test_visualizations_discovery.py tests/test_visualizations_html_parity.py
git commit -m "$(cat <<'EOF'
test(visualizations): discovery smoke + HTML parity against golden fixtures

test_visualizations_discovery.py asserts all 7 Visualization Steps appear
in core.link_registry after `import v2ecoli`.

test_visualizations_html_parity.py runs each reports/*.py against the
committed goldens in tests/fixtures/visualizations/ and structurally
diffs the HTML (BeautifulSoup tag/ID/class compare, ignoring the
run-dependent repro banner).

7 parametrized parity cases, all marked @pytest.mark.sim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 8: Produce PR-ready summary

Write a one-screen markdown summary that includes:

- Acceptance-criteria checklist (PASS/FAIL on each probe from Step 6).
- Commit list (`git log --oneline main..HEAD`).
- File-change stats (`git diff --shortstat main..HEAD`).
- Suggested PR title: `feat(visualizations): port reports to pbg-superpowers Visualization Steps`.
- Suggested PR body: include "Fixture changes" (7 new goldens), "Files removed" (v2ecoli/viz/), the new public API surface (7 Visualization classes), test plan.

Stop here. The user pushes the branch and opens the PR.

---

## Self-Review

### Spec coverage

Every spec section maps to at least one task:

| Spec section | Task(s) |
|---|---|
| Architecture: `visualizations/` subpackage parallel to `composites/` | Task 1 (scaffold) |
| Architecture: `_helpers.py` consolidation | Task 1 |
| Discovery via `bigraph_schema.package.discover` | Tasks 1, 10 |
| API contract — Visualization base, `inputs()`, `update()`, `outputs()` | Each per-Step task (2-8) |
| 3 input-shape families (structural / single / multi) | Tasks 2 + 7 (structural), 5 + 6 + 8 (single), 3 + 4 (multi) |
| `config_schema` uses dict-literal defaults | Each per-Step task (template explicit) |
| `render_document` + `render_repro_banner` scaffold | Task 1 |
| Trajectory utilities (`history_to_arrays`, etc.) | Task 1 |
| CLI wrapper pattern (`reports/*.py` thin wrapper) | Tasks 2-8 (step 6 of each) |
| Per-report variants (subprocess for benchmark/v1_v2, parallel build for compare) | Tasks 3, 4, 7 |
| Delete `v2ecoli/viz/network.py` | Task 9 |
| Update `scripts/viz_baseline.py` + `scripts/viz_network.py` | Task 9 |
| Step-level unit tests | Tasks 2-8 (each step 1) |
| HTML parity tests | Task 10 |
| Discovery smoke test | Task 10 |
| Helper-level tests | Task 1 |
| Acceptance criteria | Task 10 step 6 |
| Risks (HTML parity false positives, pbg-superpowers in dev, gather_emitter_results shape) | Tested in Task 10; flagged in PR body |

### Placeholder scan

- `_render_body()` in each per-Step task contains `raise NotImplementedError("Migrate from reports/<name>_report.py")`. This is a deliberate write-the-skeleton-then-migrate-body pattern (same as the standardize-composites plan). The migration of the actual body is part of the same step.
- No `TBD`, `TODO`, "appropriate error handling", or "similar to Task N" markers.
- Per-Step tasks describe patterns rather than enumerate every line change in the legacy report. Migration is the implementer's job; the patterns + skeletons + tests + parity gate provide the full guidance.

### Type consistency

- `class <Name>Visualization(Visualization)` — consistent across all 7 Steps.
- Each Step's `update(state) -> dict` returns `{"html": str}` — consistent.
- `inputs()` returns `dict[str, Any]` with bigraph-schema type strings — consistent.
- `_helpers.py` exports stable names: `classify`, `build_graph`, `render_cytoscape_html` (renamed from `render_html`), `render_document`, `render_repro_banner`, `history_to_arrays`, `group_by_generation`, `group_by_agent`.
- The wrapper pattern is consistent across all 7 reports: parse args → build composite(s) → run → extract trajectory → dispatch to Step → write HTML.

### Notes for the executor

- **Tasks 2-8 are independent of each other** at the framework level — each is a port. Order: 2 (smallest report, sets the pattern) → 3 → 4 → 5 → 6 → 7 → 8 (biggest). Bigger reports benefit from the patterns established by earlier tasks.
- **Migration of `_render_body`** is the bulk of each per-Step task. The plan can't enumerate hundreds of lines per report; the implementer reads the legacy file, identifies rendering vs orchestration, and moves rendering. The unit test (synthetic state) and the parity test (against golden) gate correctness.
- **Task 1's golden capture step** depends on each `reports/<name>_report.py` actually working on current main. If a report is currently broken on main (e.g., benchmark needs a specific vEcoli install), use `--only` to skip it; the corresponding parity test in Task 10 may need an `xfail` marker.
- **Task 10's parity test is the equivalence gate.** If any of the 7 fails, the corresponding `_render_body` migration is incomplete — fix it before merging.
