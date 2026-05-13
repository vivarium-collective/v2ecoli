# Port v2ecoli reports to pbg-superpowers Visualization Steps

**Status:** Design approved, awaiting implementation plan.
**Date:** 2026-05-13.
**Branch:** `port-visualizations` (cut from `main` @ `5e30aa0`).

## Goal

Port each of the 7 v2ecoli report scripts to a registered `Visualization(Step)` subclass from `pbg_superpowers.visualization`, making them discoverable by the pbg-template dashboard's Visualizations tab via `bigraph_schema.package.discover`. Each Step owns the rendering logic and produces `{'html': str}` from typed inputs; the existing `reports/*.py` CLI scripts stay at their current paths and become thin wrappers that build composites, run them, and dispatch to their Step.

## Non-goals

- Streaming-mode Visualization Steps. Every Step in this PR is invoked exactly once with a complete trajectory; per-step accumulators are a follow-up.
- Behavior changes to the underlying simulations. Reports produce HTML that's structurally equivalent to current `main`.
- A unified `v2ecoli-viz` CLI entry point. Per-report CLIs (`python reports/<name>_report.py`) keep working.
- Visualization Steps for `v2ecoli/processes/parca/viz/`. That's parca-internal and not in scope.
- Per-architecture comparison Steps that take time-series. `CompareVisualization` v0 renders side-by-side network diagrams (structural only); time-series comparison is a follow-up.

## Architecture

### Final file layout

```
v2ecoli/
├── visualizations/                # NEW subpackage, parallel to composites/
│   ├── __init__.py                # imports all 7 Steps so discovery side-effects fire
│   ├── _helpers.py                # shared infra: Cytoscape, repro_banner, HTML scaffold,
│   │                              #   trajectory utilities
│   ├── network.py                 # NetworkVisualization
│   ├── compare.py                 # CompareVisualization
│   ├── workflow.py                # WorkflowVisualization
│   ├── multigeneration.py         # MultigenerationVisualization
│   ├── colony.py                  # ColonyVisualization
│   ├── benchmark.py               # BenchmarkVisualization
│   └── v1_v2.py                   # V1V2Visualization
├── viz/                           # legacy helper — DELETED in this PR
└── ... (everything else unchanged)
```

### Deleted

- `v2ecoli/viz/network.py` — `build_graph`, `render_html`, `write_outputs`, `classify` move to `v2ecoli/visualizations/_helpers.py`.

The `v2ecoli/viz/` package directory itself goes away after that file is removed (only `__init__.py` and `network.py` exist there today).

### Untouched

- `v2ecoli/processes/parca/viz/` — parca-internal; separate concern, separate PR if ever ported.

### Discovery

`v2ecoli/visualizations/__init__.py` imports each Step module so `@composite_generator`-style decorator side-effects fire at import time. Since `Visualization` extends `Step` extends `Edge`, the subclasses auto-register into `core.link_registry` via `bigraph_schema.package.discover` whenever a consumer calls `allocate_core()`. No manual `core.register_link()` calls required.

## API contract

Every Step follows the same shape:

```python
# v2ecoli/visualizations/<name>.py
from typing import Any
from pbg_superpowers.visualization import Visualization


class <Name>Visualization(Visualization):
    """<one-line description>"""

    config_schema = {
        **Visualization.config_schema,                    # 'title': string
        # Per-viz config knobs as dict-literal defaults.
        # Use {"_type": "string", "_default": ""}, NOT "maybe[string]" —
        # the Task-4 lesson from standardize-composites: maybe[...] doesn't
        # parse in this bigraph-schema version.
    }

    def inputs(self) -> dict:
        return {
            # typed input ports in bigraph-schema vocabulary
        }

    def update(self, state: dict[str, Any]) -> dict:
        html = self._render(state)
        return {"html": html}

    # private helpers (graph extraction, HTML templating) internal to the file
```

### Three input-shape families

| Family | Steps | `inputs()` shape |
|---|---|---|
| **Structural** | `NetworkVisualization`, `CompareVisualization` | `{"composite_spec": "map[any]"}` (single) or `{"composite_specs": "list[map[any]]"}` (3-up) |
| **Single-trajectory** | `WorkflowVisualization`, `MultigenerationVisualization`, `ColonyVisualization` | `{"history": "list[map[any]]", "metadata": "map[any]"}` |
| **Multi-trajectory comparison** | `BenchmarkVisualization`, `V1V2Visualization` | one `"history_<label>": "list[map[any]]"` port per source + `"metadata": "map[any]"` |

`CompareVisualization` v0 stays structural — it renders three side-by-side network diagrams of the three v2ecoli architectures from their composite specs, no trajectory needed. Bumping it to time-series comparison is a follow-up.

### Contract specifics

1. **Return shape `{"html": str}`** is mandated by the `Visualization` base. The HTML is a complete standalone document (full `<html>...</html>`, CSS/JS inlined or via CDN). Suitable for writing to disk, serving from the dashboard, or embedding.

2. **`inputs()` is the discovery surface.** The pbg-template dashboard's Investigations runner inspects each Step's `inputs()` to wire SQLiteEmitter trajectory columns to ports.

3. **No simulation inside `update()`.** Steps are pure renderers. The caller (CLI wrapper or dashboard) builds + runs the composite separately and feeds in the result.

4. **`config_schema` uses dict-literal defaults.** Following the Task-4 lesson from `standardize-composites`: `{"_type": "string", "_default": ""}`, not `"maybe[string]"`. Confirmed parseable by `bigraph_schema.allocate_core()`.

### History row shape

The trajectory-consuming Steps assume the SQLiteEmitter trajectory comes back as `list[dict]` where each dict is one tick's wired-store state (the shape produced by `gather_emitter_results(composite)` or by querying the SQLiteEmitter directly). Steps that need it grouped differently (per-cell, per-generation) do that grouping internally via helpers in `_helpers.py`.

## Shared helpers (`v2ecoli/visualizations/_helpers.py`)

Migrates `v2ecoli/viz/network.py` and adds shared scaffolding used across the 7 Steps:

```python
# --- Cytoscape network rendering (migrated from v2ecoli/viz/network.py) ---
def classify(name: str) -> tuple[str, str]: ...
def build_graph(composite, layers) -> dict: ...          # → {nodes, edges, layers, legend}
def render_cytoscape_html(data, title, subtitle) -> str: ...

# --- HTML scaffolding shared across all 7 Steps ---
def render_repro_banner() -> str: ...                    # date + git + host + python
def render_document(title: str, body_html: str,
                    head_extra: str = "",
                    include_banner: bool = True) -> str: ...

# --- Trajectory utilities ---
def history_to_arrays(history, paths) -> dict[str, "np.ndarray"]: ...
def group_by_generation(history) -> list[list[dict]]: ...
def group_by_agent(history) -> dict[str, list[dict]]: ...
```

`render_document` is the new common scaffold — every Step's output is `render_document(title=..., body_html=<viz-specific markup>)`, so the repro banner, head metadata, and CSS reset live in one place. Each Step body just produces its content fragment.

Underscore-prefixed (`_helpers`) signals "internal to the visualizations package." `_helpers` is not exported via `__all__`. Callers outside `visualizations/` (scripts/viz_baseline.py, scripts/viz_network.py) get updated imports to point here.

## CLI wrapper pattern (`reports/*.py`)

Each report becomes a ~40-line script:

```python
# reports/<name>_report.py
"""<original docstring>"""
import argparse
import os

from process_bigraph.emitter import gather_emitter_results
from v2ecoli import build_composite
from v2ecoli.visualizations.<name> import <Name>Visualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/reports/<name>.html")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="out/cache")
    # ... viz-specific args (architecture, runtime, etc.)
    args = parser.parse_args()

    composite = build_composite("<arch>", seed=args.seed, cache_dir=args.cache_dir)
    composite.run(RUNTIME_SECONDS)
    history = gather_emitter_results(composite)

    viz = <Name>Visualization(
        config={"title": "<title>"},
        core=composite.core,
    )
    result = viz.update({"history": history, "metadata": {...}})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(result["html"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

### Per-report variants

- **`reports/network_report.py`** — skips `composite.run(...)`; passes `composite_spec` (architecture + layers) directly.
- **`reports/compare_report.py`** — builds three composites (baseline, departitioned, reconciled); passes `composite_specs=[spec_baseline, spec_departitioned, spec_reconciled]`.
- **`reports/workflow_report.py`**, **`multigeneration_report.py`**, **`colony_report.py`** — single-trajectory wrappers using the template above with their architecture / duration / colony settings.
- **`reports/benchmark_report.py`** — builds one v2ecoli composite locally + invokes one vEcoli composite via subprocess (`vEcoli[dev]` is already a runtime dep). Each writes a trajectory; the wrapper reads both back and passes them to `BenchmarkVisualization` via `history_v2ecoli=`, `history_vecoli=`.
- **`reports/v1_v2_report.py`** — three subprocesses (vEcoli 1.0, vEcoli 2.0, v2ecoli); same wrapper pattern.

Subprocess orchestration lives entirely in the wrapper, not the Step. The Step receives N pre-computed trajectories and renders.

### Other callers of the legacy helpers

- `scripts/viz_baseline.py` and `scripts/viz_network.py` currently import from `v2ecoli/viz/network.py`. They get one-line import updates to `from v2ecoli.visualizations._helpers import ...`.

## Testing

### 1. HTML equivalence parity tests (the primary gate)

`tests/test_visualizations_html_parity.py` — 7 `@pytest.mark.sim` tests, one per report. Each:
1. Builds the composite(s) the report needs (via `build_composite`).
2. Runs the sim (or just inspects the composite structure for `NetworkVisualization`).
3. Invokes the Visualization Step's `update(state)`.
4. Diffs the resulting HTML against a checked-in golden under `tests/fixtures/visualizations/<name>.golden.html`.

The diff is **structural**, not byte-for-byte:
- Parse both HTMLs with `html.parser` or `BeautifulSoup`.
- Compare element counts + attribute keys per tag.
- Compare class names and IDs (case-sensitive set equality).
- Compare embedded JSON blocks (Cytoscape graph data) for the structural reports — these are deterministic and should match byte-for-byte after canonical JSON ordering.
- Allowlist of run-dependent fragments to ignore: date strings, git SHAs, hostnames, run timestamps. These live in the repro banner and any time-of-render headers.

### 2. Step-level unit tests — `tests/test_visualizations_<name>.py`

For each Step, a `@pytest.mark.fast` test:
- Constructs the Step with a synthetic minimal config.
- Passes synthetic input matching the Step's `inputs()` schema.
- Asserts `update(state)` returns `{"html": ...}` with `"<html"` and `"</html>"` markers.
- Spot-checks one expected DOM element (chart title, section header) per Step.

No ParCa cache required. Roughly 7 unit-level tests + 1-2 helper tests for `_helpers.py` (`build_graph` extracts a known shape; `render_document` produces valid HTML wrapping).

### 3. Discovery test

`tests/test_visualizations_discovery.py` — one `@pytest.mark.fast` smoke test:

```python
def test_all_visualizations_discoverable():
    import v2ecoli  # noqa
    from bigraph_schema import allocate_core
    core = allocate_core()
    expected = {
        "NetworkVisualization", "CompareVisualization", "WorkflowVisualization",
        "MultigenerationVisualization", "ColonyVisualization",
        "BenchmarkVisualization", "V1V2Visualization",
    }
    found = {k.rsplit(".", 1)[-1] for k in core.link_registry
             if k.startswith("v2ecoli.visualizations.")}
    assert expected <= found
```

### 4. CLI wrapper smoke check

For each `reports/<name>_report.py`, a `py_compile` check in CI:

```bash
uv run python -m py_compile reports/benchmark_report.py reports/colony_report.py \
    reports/compare_report.py reports/multigeneration_report.py reports/network_report.py \
    reports/v1_v2_report.py reports/workflow_report.py
```

Same pattern as the standardize-composites refactor's script verification.

### 5. Goldens regenerator

`scripts/regenerate_viz_goldens.py` (new) — runs each Step exactly the way the parity tests would and writes the goldens into `tests/fixtures/visualizations/`. Per AGENTS.md, fixture refresh is a deliberate act called out in the PR description.

### Acceptance criteria

- [ ] `python reports/<name>_report.py` for each of the 7 produces an HTML file at `out/reports/<name>.html` that's structurally equivalent to current `main`.
- [ ] `rg "v2ecoli\\.viz\\.network|from v2ecoli\\.viz" v2ecoli/ tests/ scripts/ reports/ --type py` returns zero hits.
- [ ] All 7 `Visualization` subclasses register in `core.link_registry`.
- [ ] `pytest -m "not sim" tests/test_visualizations_*.py tests/test_visualizations_discovery.py` passes.
- [ ] `pytest -m sim tests/test_visualizations_html_parity.py` passes.
- [ ] `py_compile reports/*.py` succeeds.
- [ ] HTML output of all 7 reports rendered into `out/reports/` and attached to the PR per AGENTS.md.

## Risks

1. **HTML parity false positives.** Run-to-run timestamps (repro banner) or random-seed-dependent numeric output can produce spurious diffs. Mitigation: structural diff (DOM shape, not text content); allowlist of expected-different fragments (date / git SHA / host) explicit in the parity test.
2. **`pbg_superpowers.visualization` is "in development".** The base class or `as_visualization` decorator may evolve. We pin a lower bound on `pbg-superpowers` in `pyproject.toml` and call this out in the PR description's "Dependency changes" section.
3. **Multi-trajectory wrappers depend on external runtimes.** `benchmark_report` and `v1_v2_report` invoke vEcoli via subprocess. If the vEcoli install changes, the wrapper breaks (not the Step). The Step itself only consumes pre-computed trajectories.
4. **`gather_emitter_results()` shape.** Steps assume `list[dict]` from the emitter. Verified against current usage in `tests/_state_equal.py` and other sim tests; emitter API has been stable through the prior PRs.
5. **Goldens drift over time.** The structural-equivalence diff tolerates some drift, but a deep refactor of the rendering layer (e.g., upgrading Cytoscape.js) will require golden regeneration. The script `scripts/regenerate_viz_goldens.py` makes this a one-command operation.

## Open items (deferred)

- **Streaming-mode Visualizations** — `update(state)` called per-step in a sim composite, accumulating internally. Useful for live dashboards. Out of scope; the Step contract already supports it.
- **`v2ecoli/processes/parca/viz/`** port — parca-internal helpers; separate concern.
- **Unified `v2ecoli-viz` CLI** — one dispatch entry point instead of seven per-report scripts. Possible follow-up; AGENTS.md's CI workflow currently expects per-report paths.
- **Time-series `CompareVisualization`** — beyond structural side-by-side, render trajectory comparisons across the three architectures. Bumped from v0.

## Sequencing

**Single PR.** Estimated size:

| Bucket | Count |
|---|---|
| New files | ~10 (7 Visualization Steps + `visualizations/__init__.py` + `visualizations/_helpers.py` + `scripts/regenerate_viz_goldens.py`) |
| Deleted files | 1 (`v2ecoli/viz/network.py`) — plus the empty `v2ecoli/viz/` directory |
| Modified — `reports/*.py` | 7 (each becomes a thin wrapper) |
| Modified — internal | 2 (`scripts/viz_baseline.py`, `scripts/viz_network.py`) |
| New tests | 9 (7 step-level + 1 parity + 1 discovery) |
| Golden fixtures | 7 HTML files under `tests/fixtures/visualizations/` |

Roughly 30-35 file changes. Each Step's port is self-contained, but helpers + CLI rewrites all touch the same edges — easier to land atomically.
