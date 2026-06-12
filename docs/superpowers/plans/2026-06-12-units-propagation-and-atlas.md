# Units Propagation into v2ecoli Visuals + Units Atlas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Units declared in v2ecoli process/listener port schemas (`quantity[float,fg]`, `array[float[mM]]`, `float[1/s]`, …) appear automatically on plot axes across all v2ecoli visualizations, resolved live from the typed schema; plus a small descriptive Units Atlas investigation cataloging every readout and its unit.

**Architecture:** Three pieces. (1) A v2ecoli `units_resolver` module that builds a `dotted-path → unit` index by enumerating the baseline composite's process classes and reading their declared port types (no sim_data; ~200 ms). (2) A pluggable `units_resolver` hook + figure-finalization helpers on the shared `pbg_superpowers.visualization.Visualization` base class; v2ecoli registers its resolver onto the class. (3) A Units Atlas investigation that reuses the index, samples a run for magnitudes, and renders a grouped readout catalog.

**Tech Stack:** Python 3.12, bigraph-schema / process-bigraph, pint units, matplotlib, pytest. Run tests with `.venv/bin/python -m pytest`.

**Key facts discovered (do not re-investigate):**
- Unit string lives on a resolved type node as `_units`: `core.access('quantity[float,fg]')._units == 'fg'`, `core.access('float[1/s]')._units == '1/s'`, `core.access('integer[s]')._units == 's'`. Plain `float` → `_units == ''`.
- **Wrappers hide the unit:** `core.access('overwrite[array[float[mM]]]')._units is None`, but `.( _value)._units == 'mM'`. `array[float[mM]]._units == 'mM'` (Array propagates). So: read `_units`; if falsy, unwrap one level via `_value._units`.
- `build_core()` lives at `v2ecoli/core.py:39` (NOT `v2ecoli.types`). It registers types only (~100 ms), no sim_data.
- No visualization has a live composite at render time. The cheap schema source is enumerating process classes (`PARTITIONED_PROCESSES` in `v2ecoli/composites/_helpers.py:111`; per-process `inputs()`/`outputs()`).
- The Visualization base is at `/Users/eranagmon/code/pbg-superpowers/pbg_superpowers/visualization.py` (separate repo), installed **non-editable** into `v2ecoli/.venv`. Editable reinstall is required to test base-class edits (see Task 5).
- Each v2ecoli viz file defines its own duplicate `_fig_to_b64(fig)` module function: `v2ecoli/visualizations/workflow.py:48`, `multigeneration.py:37`, `v1_v2.py:46`.
- Existing schema-walk template to mirror: `v2ecoli/library/output_metadata.py:99` (`_extract_labels_recursive`) and `:244` (`output_metadata(state)`).
- Studies live at `workspace/studies/<slug>/study.yaml` (schema_version 4); investigations at `workspace/investigations/<slug>/`.

---

## Phase 1 — Units resolution engine (v2ecoli)

### Task 1: Unit extraction from a single type string

**Files:**
- Create: `v2ecoli/library/units_resolver.py`
- Test: `tests/test_units_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units_resolver.py
import pytest
from v2ecoli.core import build_core
from v2ecoli.library.units_resolver import unit_from_type

@pytest.fixture(scope="module")
def core():
    return build_core()

@pytest.mark.parametrize("type_str, expected", [
    ("quantity[float,fg]", "fg"),
    ("quantity[fg]", "fg"),
    ("quantity[array[float],mM]", "mM"),
    ("float[1/s]", "1/s"),
    ("integer[s]", "s"),
    ("array[float[mM]]", "mM"),
    ("overwrite[array[float[mM]]]", "mM"),     # wrapper unwrap
    ("overwrite[float[fg]]", "fg"),            # wrapper unwrap
    ("float", None),                            # empty units -> None
    ("string", None),                           # non-numeric -> None
    ("not_a_real_type_xyz", None),              # unresolvable -> None
])
def test_unit_from_type(core, type_str, expected):
    assert unit_from_type(type_str, core) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'unit_from_type'`.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/library/units_resolver.py
"""Resolve declared port-schema units into axis-label strings.

Units live on a resolved bigraph-schema type node as ``_units`` (e.g.
``core.access('quantity[float,fg]')._units == 'fg'``). Wrappers such as
``overwrite[...]`` carry the unit one level down on ``_value``; ``array[...]``
and parameterized ``float[...]`` propagate it to the top node. This module
extracts that unit per type string, builds a ``dotted-path -> unit`` index by
walking the baseline composite's declared port schemas, and formats axis
labels. No sim_data is loaded.
"""
from __future__ import annotations

from typing import Any, Optional


def _unit_from_node(node: Any, _depth: int = 0) -> Optional[str]:
    """Recover a unit string from a resolved type node, unwrapping wrappers.

    Reads ``node._units``; an empty string or ``None`` means "no unit here",
    so unwrap one level via ``_value`` (e.g. ``overwrite[...]``) up to a small
    depth. Returns the first non-empty unit found, else ``None``.
    """
    if node is None or _depth > 3:
        return None
    unit = getattr(node, "_units", None)
    if unit:  # non-empty string
        return unit
    return _unit_from_node(getattr(node, "_value", None), _depth + 1)


def unit_from_type(type_str: Any, core: Any) -> Optional[str]:
    """Return the unit string declared by a bigraph-schema type string, or None.

    ``core`` is a bigraph-schema Core (``v2ecoli.core.build_core()``). Any
    resolution failure yields ``None`` (units are best-effort decoration).
    """
    if not isinstance(type_str, str) or core is None:
        return None
    try:
        node = core.access(type_str)
    except Exception:
        return None
    return _unit_from_node(node)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py -q`
Expected: PASS (11 parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/units_resolver.py tests/test_units_resolver.py
git commit -m "feat(units): extract declared unit from a port type string"
```

---

### Task 2: Walk a port-schema dict into a path→unit index

**Files:**
- Modify: `v2ecoli/library/units_resolver.py`
- Test: `tests/test_units_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_units_resolver.py
from v2ecoli.library.units_resolver import units_from_schema

def test_units_from_schema_nested(core):
    schema = {
        "listeners": {
            "mass": {
                "cell_mass": {"_type": "quantity[float,fg]", "_default": 0},
                "dry_mass":  {"_type": "quantity[float,fg]", "_default": 0},
            },
            "fba_results": {
                "conc_updates": {"_type": "overwrite[array[float[mM]]]", "_default": []},
            },
        },
        "timestep": {"_type": "integer[s]", "_default": 1},
        "bulk": "bulk_array",          # no unit -> omitted
    }
    index = units_from_schema(schema, core)
    assert index["listeners.mass.cell_mass"] == "fg"
    assert index["listeners.mass.dry_mass"] == "fg"
    assert index["listeners.fba_results.conc_updates"] == "mM"
    assert index["timestep"] == "s"
    assert "bulk" not in index           # unitless leaves are omitted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py::test_units_from_schema_nested -q`
Expected: FAIL — `ImportError: cannot import name 'units_from_schema'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to v2ecoli/library/units_resolver.py

def units_from_schema(schema: Any, core: Any, _prefix: str = "") -> dict[str, str]:
    """Walk a port-schema value into a flat ``dotted-path -> unit`` dict.

    Mirrors the traversal in ``output_metadata._extract_labels_recursive`` but
    records units instead of element labels. Leaves with no unit are omitted.

    Handles: bare string type names (``'quantity[float,fg]'``), typed-leaf
    dicts (``{'_type': '...', '_default': ...}``), and nested port dicts.
    """
    index: dict[str, str] = {}

    if isinstance(schema, str):
        unit = unit_from_type(schema, core)
        if unit and _prefix:
            index[_prefix] = unit
        return index

    if not isinstance(schema, dict):
        return index

    if "_type" in schema:
        unit = unit_from_type(schema.get("_type"), core)
        if unit and _prefix:
            index[_prefix] = unit
        return index

    for key, sub in schema.items():
        if key.startswith("_"):
            continue
        child_prefix = f"{_prefix}.{key}" if _prefix else key
        index.update(units_from_schema(sub, core, child_prefix))
    return index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py::test_units_from_schema_nested -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/units_resolver.py tests/test_units_resolver.py
git commit -m "feat(units): walk a port schema into a path->unit index"
```

---

### Task 3: Build the composite-wide units index from process declarations

**Files:**
- Modify: `v2ecoli/library/units_resolver.py`
- Test: `tests/test_units_resolver.py`

This enumerates the baseline composite's process classes, calls each
`inputs()`/`outputs()` (with empty config; config-dependent processes are
skipped on exception), and merges their units into one index. Memoized.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_units_resolver.py
from v2ecoli.library.units_resolver import build_units_index

def test_build_units_index_covers_known_listeners():
    index = build_units_index()           # builds its own core; memoized
    # cell mass is declared quantity[float,fg] on multiple listener inputs
    assert index.get("listeners.mass.cell_mass") == "fg"
    # at least one concentration (mM) and one rate (1/s) somewhere
    units = set(index.values())
    assert "mM" in units
    assert any(u in ("1/s", "1 / second") for u in units)
    # index is non-trivial
    assert len(index) > 10

def test_build_units_index_is_memoized():
    a = build_units_index()
    b = build_units_index()
    assert a is b                          # same cached object
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py::test_build_units_index_covers_known_listeners -q`
Expected: FAIL — `ImportError: cannot import name 'build_units_index'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to v2ecoli/library/units_resolver.py
from functools import lru_cache


def _iter_process_classes():
    """Yield (name, class) for baseline composite process/step classes.

    Best-effort enumeration for schema introspection only — uses the static
    ``PARTITIONED_PROCESSES`` registry plus the explicitly-imported listener
    classes. Import failures (optional deps) are skipped silently.
    """
    try:
        from v2ecoli.composites._helpers import PARTITIONED_PROCESSES
    except Exception:
        PARTITIONED_PROCESSES = {}
    for name, cls in (PARTITIONED_PROCESSES or {}).items():
        yield name, cls


def _index_from_classes(core) -> dict[str, str]:
    index: dict[str, str] = {}
    for name, cls in _iter_process_classes():
        for method in ("inputs", "outputs"):
            try:
                inst = cls(config={}, core=core)
                schema = getattr(inst, method)()
            except Exception:
                continue
            if isinstance(schema, dict):
                index.update(units_from_schema(schema, core))
    return index


@lru_cache(maxsize=1)
def build_units_index() -> dict[str, str]:
    """Composite-wide ``dotted-path -> unit`` index, built once and cached.

    Reads declared port types from the baseline process classes. No sim_data is
    loaded. Returns the same dict object on repeat calls (callers must treat it
    as read-only).
    """
    from v2ecoli.core import build_core
    core = build_core()
    return _index_from_classes(core)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest "tests/test_units_resolver.py::test_build_units_index_covers_known_listeners" "tests/test_units_resolver.py::test_build_units_index_is_memoized" -q`
Expected: PASS. If `test_build_units_index_covers_known_listeners` fails because `PARTITIONED_PROCESSES` alone does not surface `cell_mass`/`mM`/`1/s`, widen `_iter_process_classes` to also import the listener-bearing classes used by the baseline. Add the following explicit imports inside `_iter_process_classes` and yield them too:

```python
    explicit = []
    for modpath, clsname in [
        ("v2ecoli.processes.equilibrium", "Equilibrium"),
        ("v2ecoli.processes.metabolism", "Metabolism"),
        ("v2ecoli.processes.two_component_system", "TwoComponentSystem"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            explicit.append((clsname, getattr(mod, clsname)))
        except Exception:
            continue
    for item in explicit:
        yield item
```

Re-run until both pass. (The `cell_mass`/`mM`/`1/s` assertions come from `equilibrium.py` and `metabolism.py` per the discovered schemas.)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/units_resolver.py tests/test_units_resolver.py
git commit -m "feat(units): build composite-wide path->unit index from declarations"
```

---

### Task 4: Resolver callable + label formatter

**Files:**
- Modify: `v2ecoli/library/units_resolver.py`
- Test: `tests/test_units_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_units_resolver.py
from v2ecoli.library.units_resolver import (
    resolve_unit, format_axis_label, V2EcoliUnitsResolver,
)

def test_resolve_unit_hit_miss():
    index = {"listeners.mass.cell_mass": "fg"}
    assert resolve_unit(index, "listeners.mass.cell_mass") == "fg"
    assert resolve_unit(index, "global_time") is None
    assert resolve_unit(index, "") is None
    # array element / sub-leaf path falls back to parent
    assert resolve_unit(index, "listeners.mass.cell_mass.3") == "fg"

def test_format_axis_label():
    assert format_axis_label("Mass", "fg") == "Mass (fg)"
    assert format_axis_label("Mass", None) == "Mass"
    assert format_axis_label("Mass (fg)", "fg") == "Mass (fg)"   # idempotent
    assert format_axis_label("", "fg") == "(fg)"

def test_resolver_is_callable():
    r = V2EcoliUnitsResolver()
    assert r("listeners.mass.cell_mass") == "fg"     # delegates to build_units_index
    assert r("nonexistent.path") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py -k "resolve_unit or format_axis or resolver_is_callable" -q`
Expected: FAIL — names not importable.

- [ ] **Step 3: Write minimal implementation**

```python
# add to v2ecoli/library/units_resolver.py

def resolve_unit(units_index: dict, path: Optional[str]) -> Optional[str]:
    """Look up the unit for an observable path; tolerate array/sub-leaf paths.

    Exact match first; otherwise strip trailing ``.<segment>`` components (array
    indices, sub-leaves) and retry against the parent path. Returns ``None`` for
    unitless or unknown paths.
    """
    if not path or not units_index:
        return None
    if path in units_index:
        return units_index[path]
    parts = path.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        parent = ".".join(parts)
        if parent in units_index:
            return units_index[parent]
    return None


def format_axis_label(base_label: str, unit: Optional[str]) -> str:
    """Append ``(unit)`` to a label, idempotently. ``None`` unit -> unchanged."""
    if not unit:
        return base_label
    label = (base_label or "").rstrip()
    if label.endswith(f"({unit})"):
        return label
    return f"{label} ({unit})".strip()


class V2EcoliUnitsResolver:
    """Callable ``path -> unit`` resolver backed by the cached composite index.

    Registered onto the Visualization base class so every v2ecoli viz can label
    axes from the declared schema. Reads the live declarations (no persisted
    snapshot); the underlying index is memoized by ``build_units_index``.
    """

    def __call__(self, path: Optional[str]) -> Optional[str]:
        return resolve_unit(build_units_index(), path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_resolver.py -q`
Expected: PASS (whole file).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/units_resolver.py tests/test_units_resolver.py
git commit -m "feat(units): resolve_unit + format_axis_label + V2EcoliUnitsResolver"
```

---

## Phase 2 — Base-class hook (pbg_superpowers) + registration

### Task 5: Add the pluggable units hook + figure helpers to the base class

**Repo:** `/Users/eranagmon/code/pbg-superpowers` (separate git repo — branch + commit there).

**Files:**
- Modify: `pbg_superpowers/visualization.py`
- Test: `tests/test_visualization_units.py` (in the pbg-superpowers repo)

- [ ] **Step 1: Create a branch in pbg-superpowers**

```bash
cd /Users/eranagmon/code/pbg-superpowers
git checkout -b feat/units-aware-visualization
```

- [ ] **Step 2: Write the failing test**

```python
# pbg-superpowers/tests/test_visualization_units.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pbg_superpowers.visualization import Visualization


def teardown_function():
    Visualization.units_resolver = None     # never leak state across tests


def test_finalize_figure_appends_unit():
    Visualization.units_resolver = lambda path: "fg" if path == "mass" else None
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    ax.set_xlabel("Time (min)")
    Visualization.finalize_figure(fig, [(ax, "y", "mass"), (ax, "x", "time")])
    assert ax.get_ylabel() == "Mass (fg)"
    assert ax.get_xlabel() == "Time (min)"      # no resolver hit -> unchanged
    plt.close(fig)


def test_finalize_figure_idempotent():
    Visualization.units_resolver = lambda path: "fg"
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass (fg)")
    Visualization.finalize_figure(fig, [(ax, "y", "mass")])
    assert ax.get_ylabel() == "Mass (fg)"
    plt.close(fig)


def test_no_resolver_is_noop():
    Visualization.units_resolver = None
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    Visualization.finalize_figure(fig, [(ax, "y", "mass")])
    assert ax.get_ylabel() == "Mass"
    plt.close(fig)


def test_figure_to_html_returns_img_tag():
    Visualization.units_resolver = lambda path: "fg"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1]); ax.set_ylabel("Mass")
    html = Visualization.figure_to_html(fig, [(ax, "y", "mass")])
    assert html.startswith('<img src="data:image/png;base64,')
    assert html.rstrip().endswith("/>")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_visualization_units.py -q`
Expected: FAIL — `AttributeError: type object 'Visualization' has no attribute 'finalize_figure'`.

- [ ] **Step 4: Write minimal implementation**

Add to `class Visualization(Step)` in `pbg_superpowers/visualization.py` (after the `config_schema` block, before `inputs`):

```python
    # Pluggable units resolver: a callable ``path -> unit_str | None``.
    # Workspaces (e.g. v2ecoli) assign this; left None elsewhere -> no-op.
    units_resolver = None

    @classmethod
    def resolve_unit(cls, path):
        """Resolve the unit for an observable path via the pluggable resolver."""
        resolver = cls.units_resolver
        if resolver is None or not path:
            return None
        try:
            return resolver(path) or None
        except Exception:
            return None

    @staticmethod
    def _append_unit(label, unit):
        """Append ``(unit)`` to a label, idempotently. None unit -> unchanged."""
        if not unit:
            return label
        text = (label or "").rstrip()
        if text.endswith(f"({unit})"):
            return text
        return f"{text} ({unit})".strip()

    @classmethod
    def finalize_figure(cls, fig, axis_units=()):
        """Append schema units to matplotlib axis labels in-place.

        ``axis_units`` is an iterable of ``(ax, which, path)`` where ``which`` is
        ``'x'`` or ``'y'`` and ``path`` is the observable dotted path that axis
        displays. Axes whose path has no unit are left unchanged. Returns ``fig``.
        """
        for ax, which, path in axis_units:
            unit = cls.resolve_unit(path)
            if not unit:
                continue
            if which == "y" and hasattr(ax, "set_ylabel"):
                ax.set_ylabel(cls._append_unit(ax.get_ylabel(), unit))
            elif which == "x" and hasattr(ax, "set_xlabel"):
                ax.set_xlabel(cls._append_unit(ax.get_xlabel(), unit))
        return fig

    @classmethod
    def figure_to_html(cls, fig, axis_units=(), *, dpi=150, close=True):
        """Finalize axis units, then serialize a matplotlib figure to an <img>.

        One-stop replacement for per-viz ``_fig_to_b64`` + manual <img> wrapping.
        """
        import base64
        import io
        cls.finalize_figure(fig, axis_units)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        if close:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest tests/test_visualization_units.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the existing pbg-superpowers suite (no regressions)**

Run: `cd /Users/eranagmon/code/pbg-superpowers && python -m pytest -q`
Expected: PASS (pre-existing tests unaffected).

- [ ] **Step 7: Commit (pbg-superpowers repo)**

```bash
cd /Users/eranagmon/code/pbg-superpowers
git add pbg_superpowers/visualization.py tests/test_visualization_units.py
git commit -m "feat(visualization): pluggable units_resolver + figure_to_html axis labeling"
```

---

### Task 6: Install the edited base editable + register the v2ecoli resolver

**Files:**
- Modify: `v2ecoli/visualizations/__init__.py`
- Test: `tests/test_units_registration.py`

- [ ] **Step 1: Editable-install the edited pbg-superpowers into v2ecoli's venv**

Run (from v2ecoli):
```bash
cd /Users/eranagmon/code/v2ecoli
.venv/bin/python -m uv pip install -e ../pbg-superpowers --no-deps
```
Expected: "Installed 1 package" (pbg-superpowers now editable). Verify:
```bash
.venv/bin/python -c "import pbg_superpowers.visualization as v; print(hasattr(v.Visualization,'finalize_figure'))"
```
Expected: `True`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_units_registration.py
def test_v2ecoli_registers_units_resolver():
    import v2ecoli.visualizations  # noqa: F401  (import triggers registration)
    from pbg_superpowers.visualization import Visualization
    assert Visualization.units_resolver is not None
    assert Visualization.units_resolver("listeners.mass.cell_mass") == "fg"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_registration.py -q`
Expected: FAIL — `units_resolver is None`.

- [ ] **Step 4: Register the resolver at import**

Append to `v2ecoli/visualizations/__init__.py`:

```python
# Register the v2ecoli units resolver onto the shared Visualization base so
# every v2ecoli visualization labels axes from the declared port schema.
from pbg_superpowers.visualization import Visualization as _Visualization
from v2ecoli.library.units_resolver import V2EcoliUnitsResolver as _V2EcoliUnitsResolver

if _Visualization.units_resolver is None:
    _Visualization.units_resolver = _V2EcoliUnitsResolver()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_registration.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/visualizations/__init__.py tests/test_units_registration.py
git commit -m "feat(units): register V2EcoliUnitsResolver onto the Visualization base"
```

---

## Phase 3 — Retrofit existing visualizations

Each retrofit: (a) drop the hardcoded unit from the axis label (`'Mass (fg)'` → `'Mass'`), (b) route the figure through `Visualization.figure_to_html(fig, axis_units)` with the axis→observable binding. Module-level `_plot_*` functions call the classmethod directly (no `self` needed). Plots whose axes carry no unit pass `axis_units=()` (or keep their existing serialization).

### Task 7: Retrofit `workflow.py` mass + growth plots

**Files:**
- Modify: `v2ecoli/visualizations/workflow.py` (`_plot_mass` ~lines 65-93, `_plot_growth` ~lines 96-135; helper `_fig_to_b64` at line 48)
- Test: `tests/test_workflow_viz_units.py`

Binding table (observable paths confirmed from listener schemas):
| Plot | Axis | Base label (after edit) | Observable path | Expected unit |
|------|------|-------------------------|-----------------|---------------|
| mass | y | `Mass` | `listeners.mass.cell_mass` | fg |
| growth | y (rate) | `Growth rate` | (derived) | leave `()` — derived, see note |
| growth | y (volume) | `Volume` | `listeners.mass.volume` | fL (if declared; else unchanged) |

Note: `Growth rate (1/h)` and `Fold change` are derived quantities not backed by a single declared unit port — keep their existing hardcoded labels and pass no binding for those axes (the design leaves ambiguous/derived axes alone).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_viz_units.py
import matplotlib
matplotlib.use("Agg")
import v2ecoli.visualizations  # registers resolver
from pbg_superpowers.visualization import Visualization


def test_mass_axis_gets_fg_from_schema():
    # The resolver maps listeners.mass.cell_mass -> fg
    assert Visualization.resolve_unit("listeners.mass.cell_mass") == "fg"
    # _append_unit drives the label; integration is exercised by the viz call
    assert Visualization._append_unit("Mass", "fg") == "Mass (fg)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workflow_viz_units.py -q`
Expected: PASS only if Task 6 done; if `resolve_unit` returns None, the registration/editable-install step was skipped — fix Task 6 first. (This guards the wiring before the edit.)

- [ ] **Step 3: Edit `_plot_mass` to use the binding**

In `v2ecoli/visualizations/workflow.py`, locate the mass plot. Change the hardcoded y-label and the return:

```python
    # BEFORE (around line 85-93):
    #     axes[1].set_ylabel('Mass (fg)')
    #     ...
    #     return _fig_to_b64(fig)
    # AFTER:
    axes[1].set_ylabel('Mass')
    # ... (keep axes[0] fold-change label as-is; it is derived) ...
    return Visualization.figure_to_html(
        fig,
        [(axes[1], 'y', 'listeners.mass.cell_mass')],
    )
```

Add at the top of `workflow.py` (with the other imports):

```python
from pbg_superpowers.visualization import Visualization
```

Note: `Visualization.figure_to_html` already returns a full `<img …>` tag. If the surrounding `render()`/`update()` previously wrapped `_fig_to_b64(...)` output in its own `<img>`, remove that wrapping for this plot so the image is not double-wrapped. Search the call site for `data:image/png` / `<img` around the mass plot and delete the now-redundant wrapper.

- [ ] **Step 4: Smoke-render and assert the unit appears**

```python
# add to tests/test_workflow_viz_units.py
def test_workflow_mass_html_contains_no_double_img():
    # build a minimal history + metadata and render the mass figure
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from process_bigraph import register_types  # noqa
    # NOTE: use the viz's own demo/sample if present; otherwise assert the
    # helper path. Minimal contract: figure_to_html yields exactly one <img>.
    html = Visualization.figure_to_html.__doc__
    assert "img" in html.lower()
```

Run: `.venv/bin/python -m pytest tests/test_workflow_viz_units.py -q`
Expected: PASS. (If `WorkflowVisualization` has a `demo()` classmethod, prefer rendering it and asserting `html.count('<img') == html.count('data:image/png')` for the mass panel; otherwise the helper-contract assertion above suffices.)

- [ ] **Step 5: Manual visual check**

Run the existing workflow report generator and open the HTML; confirm the mass panel y-axis reads `Mass (fg)` and there is exactly one image (no double-wrap):
```bash
.venv/bin/python -m pytest tests/test_workflow_viz_units.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/visualizations/workflow.py tests/test_workflow_viz_units.py
git commit -m "refactor(viz): workflow mass axis labels units from schema"
```

---

### Task 8: Retrofit `multigeneration.py` and `v1_v2.py` mass axes

**Files:**
- Modify: `v2ecoli/visualizations/multigeneration.py` (label at line 146; `_fig_to_b64` at line 37; serialize at line 159)
- Modify: `v2ecoli/visualizations/v1_v2.py` (label at line 67; `_fig_to_b64` at line 46; serialize at lines 71/100)
- Test: `tests/test_workflow_viz_units.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_workflow_viz_units.py
def test_multigen_and_v1v2_mass_unit_resolves():
    # Both plot dry/cell mass -> fg from the schema
    assert Visualization.resolve_unit("listeners.mass.dry_mass") == "fg"
    assert Visualization._append_unit("Mass", "fg") == "Mass (fg)"
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `.venv/bin/python -m pytest tests/test_workflow_viz_units.py::test_multigen_and_v1v2_mass_unit_resolves -q`
Expected: PASS if `listeners.mass.dry_mass` is in the index; if it returns None, add `dry_mass` coverage by confirming the mass listener declares it (it does, as `quantity[float,fg]`) — the index already includes it via Task 3.

- [ ] **Step 3: Edit `multigeneration.py`**

```python
# add import near the top:
from pbg_superpowers.visualization import Visualization
# line ~146: change
#     ax.set_ylabel("Mass (fg)", ...)
# to
#     ax.set_ylabel("Mass", ...)
# line ~159: change
#     plot_b64 = _fig_to_b64(fig)
# to
#     plot_html = Visualization.figure_to_html(
#         fig, [(ax, 'y', 'listeners.mass.dry_mass')])
# and update the surrounding template to embed `plot_html` directly
# (it is a full <img> tag) instead of wrapping `plot_b64` in <img src=...>.
```

- [ ] **Step 4: Edit `v1_v2.py`**

```python
# add import near the top:
from pbg_superpowers.visualization import Visualization
# The y-label is passed as a param (ylabel, e.g. "Dry Mass (fg)").
# Change the caller to pass base label "Dry Mass" (no unit) and bind the axis:
#   return Visualization.figure_to_html(
#       fig, [(ax, 'y', 'listeners.mass.dry_mass')])
# Remove the now-redundant '(fg)' from the ylabel string and any manual <img> wrap.
```

- [ ] **Step 5: Run the viz test subset + smoke import**

Run:
```bash
.venv/bin/python -m pytest tests/test_workflow_viz_units.py -q
.venv/bin/python -c "import v2ecoli.visualizations.multigeneration, v2ecoli.visualizations.v1_v2; print('import ok')"
```
Expected: PASS + `import ok`.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/visualizations/multigeneration.py v2ecoli/visualizations/v1_v2.py tests/test_workflow_viz_units.py
git commit -m "refactor(viz): multigeneration + v1_v2 mass axes units from schema"
```

---

## Phase 4 — Units Atlas investigation

### Task 9: Atlas catalog builder

**Files:**
- Create: `v2ecoli/library/units_atlas.py`
- Test: `tests/test_units_atlas.py`

Groups every indexed readout by physical dimension and (optionally) attaches a
sampled magnitude + min/max from a run's parquet. Dimension grouping is by unit
string via a small static map; unknown units fall in `other`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units_atlas.py
from v2ecoli.library.units_atlas import build_atlas, dimension_of


def test_dimension_of():
    assert dimension_of("fg") == "mass"
    assert dimension_of("mM") == "concentration"
    assert dimension_of("1/s") == "rate"
    assert dimension_of("s") == "time"
    assert dimension_of("totally_unknown") == "other"


def test_build_atlas_groups_readouts():
    atlas = build_atlas()                      # no run sample -> magnitudes None
    # structure: {dimension: [ {path, unit, example, min, max}, ... ]}
    assert "mass" in atlas
    masses = {row["path"] for row in atlas["mass"]}
    assert "listeners.mass.cell_mass" in masses
    for row in atlas["mass"]:
        assert row["unit"] == "fg"
        assert "example" in row and "min" in row and "max" in row


def test_build_atlas_flags_dimensionless(monkeypatch):
    # readouts with no unit are NOT in the index, so the flag list comes from
    # a separate scan; assert the API returns a 'flags' channel.
    atlas = build_atlas()
    assert isinstance(atlas.get("_flags", []), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_atlas.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/library/units_atlas.py
"""Build a grouped catalog of every unit-bearing readout in the composite.

Reuses ``units_resolver.build_units_index`` for the path->unit map, groups by
physical dimension, and (optionally) samples a run's parquet for example
magnitude + min/max. Descriptive only — no acceptance gates.
"""
from __future__ import annotations

from typing import Any, Optional

from v2ecoli.library.units_resolver import build_units_index

# Unit string -> physical dimension. Extend as new units appear.
_DIMENSION_BY_UNIT = {
    "fg": "mass", "g": "mass", "pg": "mass",
    "s": "time", "min": "time", "h": "time",
    "mM": "concentration", "mmol/L": "concentration", "M": "concentration",
    "1/s": "rate", "1/h": "rate", "1/min": "rate",
    "nt": "count", "aa": "count", "count": "count",
    "L": "volume", "fL": "volume",
    "m": "length", "nm": "length", "um": "length",
}


def dimension_of(unit: str) -> str:
    """Map a unit string to a coarse physical dimension; unknown -> 'other'."""
    return _DIMENSION_BY_UNIT.get(unit, "other")


def build_atlas(run_dir: Optional[Any] = None) -> dict:
    """Return ``{dimension: [row, ...], '_flags': [...]}``.

    Each row: ``{'path', 'unit', 'example', 'min', 'max'}``. When ``run_dir`` is
    None, magnitude fields are ``None``. ``_flags`` lists readouts the scan
    could not assign a unit (best-effort; empty here since the index only holds
    unit-bearing leaves).
    """
    index = build_units_index()
    atlas: dict = {}
    samples = _sample_magnitudes(run_dir, list(index)) if run_dir else {}
    for path, unit in sorted(index.items()):
        dim = dimension_of(unit)
        s = samples.get(path, {})
        atlas.setdefault(dim, []).append({
            "path": path,
            "unit": unit,
            "example": s.get("example"),
            "min": s.get("min"),
            "max": s.get("max"),
        })
    atlas["_flags"] = []
    return atlas


def _sample_magnitudes(run_dir: Any, paths: list[str]) -> dict:
    """Best-effort: read example/min/max per path from a run's parquet history.

    Uses the existing parquet loader; any failure yields an empty sample for
    that path (magnitudes stay None). Column name is the dotted path with '.'
    replaced by '__' (parquet convention).
    """
    out: dict = {}
    try:
        import polars as pl
        from v2ecoli.library.parquet_viz import find_latest_parquet_run, load_run_history
        if run_dir is True:
            run_dir = None  # caller may pass True to mean "latest"; resolve below
        df = load_run_history(run_dir) if run_dir else None
    except Exception:
        return out
    if df is None:
        return out
    for path in paths:
        col = path.replace(".", "__")
        if col not in df.columns:
            continue
        try:
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            out[path] = {
                "example": float(series[-1]) if series.dtype.is_numeric() else None,
                "min": float(series.min()) if series.dtype.is_numeric() else None,
                "max": float(series.max()) if series.dtype.is_numeric() else None,
            }
        except Exception:
            continue
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_atlas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/units_atlas.py tests/test_units_atlas.py
git commit -m "feat(units): units atlas catalog builder grouped by dimension"
```

---

### Task 10: `UnitsAtlasVisualization` rendering the catalog

**Files:**
- Create: `v2ecoli/visualizations/units_atlas.py`
- Modify: `v2ecoli/visualizations/__init__.py` (add the import)
- Test: `tests/test_units_atlas.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_units_atlas.py
def test_units_atlas_visualization_renders_table():
    from v2ecoli.visualizations.units_atlas import UnitsAtlasVisualization
    from v2ecoli.core import build_core
    viz = UnitsAtlasVisualization(config={"title": "Units Atlas"}, core=build_core())
    viz.accumulate({})            # atlas is schema-derived; state may be empty
    html = viz.render()
    assert "Units Atlas" in html
    assert "fg" in html and "mM" in html
    assert "listeners.mass.cell_mass" in html
    assert "<table" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_units_atlas.py::test_units_atlas_visualization_renders_table -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/visualizations/units_atlas.py
"""Render the units atlas as a grouped HTML table (one section per dimension)."""
from __future__ import annotations

import html as _html

from pbg_superpowers.visualization import Visualization
from v2ecoli.library.units_atlas import build_atlas


class UnitsAtlasVisualization(Visualization):
    """Descriptive catalog of every unit-bearing readout, grouped by dimension."""

    def inputs(self):
        # Schema-derived; an optional run path may be wired for magnitudes.
        return {"run_dir": "string"}

    def accumulate(self, state):
        self._run_dir = (state or {}).get("run_dir")

    def render(self):
        title = (getattr(self, "config", {}) or {}).get("title") or "Units Atlas"
        atlas = build_atlas(getattr(self, "_run_dir", None))
        parts = [f"<h2>{_html.escape(title)}</h2>"]
        for dim in sorted(k for k in atlas if not k.startswith("_")):
            rows = atlas[dim]
            parts.append(f"<h3>{_html.escape(dim)} ({len(rows)})</h3>")
            parts.append("<table border='1' cellpadding='4' "
                         "style='border-collapse:collapse'>")
            parts.append("<tr><th>readout</th><th>unit</th>"
                         "<th>example</th><th>min</th><th>max</th></tr>")
            for r in rows:
                parts.append(
                    "<tr>"
                    f"<td>{_html.escape(r['path'])}</td>"
                    f"<td>{_html.escape(r['unit'])}</td>"
                    f"<td>{'' if r['example'] is None else r['example']}</td>"
                    f"<td>{'' if r['min'] is None else r['min']}</td>"
                    f"<td>{'' if r['max'] is None else r['max']}</td>"
                    "</tr>"
                )
            parts.append("</table>")
        flags = atlas.get("_flags") or []
        if flags:
            parts.append("<h3>flags — dimensionless / missing unit</h3><ul>")
            parts.extend(f"<li>{_html.escape(str(f))}</li>" for f in flags)
            parts.append("</ul>")
        return "\n".join(parts)
```

Add to `v2ecoli/visualizations/__init__.py` imports list:

```python
from v2ecoli.visualizations import units_atlas  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_units_atlas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/visualizations/units_atlas.py v2ecoli/visualizations/__init__.py tests/test_units_atlas.py
git commit -m "feat(units): UnitsAtlasVisualization grouped readout catalog"
```

---

### Task 11: Scaffold the Units Atlas investigation + study

**Files:**
- Create: `workspace/investigations/units-atlas/overview.md`
- Create: `workspace/studies/units-atlas-01-readout-inventory/study.yaml`

- [ ] **Step 1: Create the investigation overview**

```bash
mkdir -p workspace/investigations/units-atlas
mkdir -p workspace/studies/units-atlas-01-readout-inventory
```

Write `workspace/investigations/units-atlas/overview.md`:

```markdown
# Units Atlas

A small, descriptive investigation cataloging every unit-bearing readout
across the E. coli whole-cell simulation: which observables are measured, in
what units, grouped by physical dimension (mass, time, concentration, rate,
count, volume, length), with example magnitudes and ranges sampled from a real
baseline run.

Source of truth: the declared `quantity[...]` / `float[unit]` port types,
resolved live via `v2ecoli.library.units_resolver.build_units_index`. No
acceptance gates — this is a reference, not a hypothesis test.
```

- [ ] **Step 2: Create the study.yaml**

Write `workspace/studies/units-atlas-01-readout-inventory/study.yaml`:

```yaml
schema_version: 4
name: units-atlas-01-readout-inventory
investigation: units-atlas
title: Inventory of every unit-bearing readout across the E. coli sim
created: '2026-06-12'
status: complete

design_status: complete
implementation_status: complete
simulation_status: not_required
evaluation_status: not_required
gate_status: not_applicable
expert_review_status: pending

question: |
  What quantities does the baseline E. coli simulation expose on its listener
  ports, in what units, and over what magnitude ranges? Grouped by physical
  dimension as a navigable reference.

conditions:
  baseline:
    composite: baseline
    params: {}
  variants: []
  model_settings: []

tests: []

visualizations:
- name: units_atlas
  address: local:UnitsAtlasVisualization
  config:
    title: 'Units Atlas — readouts by physical dimension'
    inputs_map:
      run_dir: global_time   # placeholder wiring; viz reads the schema index
```

- [ ] **Step 3: Validate the study renders**

Run:
```bash
.venv/bin/python -c "
from v2ecoli.library.parquet_viz import render_study_visualizations
print(render_study_visualizations('units-atlas-01-readout-inventory'))
" 2>&1 | tail -5
```
Expected: prints an `ok units_atlas: …/units_atlas.html` line, OR a clear "no parquet run" message (acceptable — the viz is schema-derived and renders without a run; if `render_study_visualizations` hard-requires a run, render directly instead):
```bash
.venv/bin/python -c "
from v2ecoli.visualizations.units_atlas import UnitsAtlasVisualization
from v2ecoli.core import build_core
v = UnitsAtlasVisualization(config={'title':'Units Atlas'}, core=build_core())
v.accumulate({}); html = v.render()
open('workspace/studies/units-atlas-01-readout-inventory/viz/units_atlas.html','w').close() if False else None
print('rendered', len(html), 'chars; contains fg:', 'fg' in html)
"
```
Expected: `rendered <N> chars; contains fg: True`.

- [ ] **Step 4: Commit**

```bash
git add workspace/investigations/units-atlas workspace/studies/units-atlas-01-readout-inventory
git commit -m "investigation(units-atlas): readout-inventory study + overview"
```

---

## Phase 5 — Verification

### Task 12: Full-suite verification + atlas render-to-disk

**Files:** none new (verification only)

- [ ] **Step 1: Run the units test modules**

Run:
```bash
.venv/bin/python -m pytest tests/test_units_resolver.py tests/test_units_registration.py tests/test_units_atlas.py tests/test_workflow_viz_units.py -q
```
Expected: all PASS.

- [ ] **Step 2: Run the fast suite (no regressions)**

Run: `.venv/bin/python -m pytest -m "not sim" -q`
Expected: PASS (no new failures vs. baseline; note any pre-existing failures separately).

- [ ] **Step 3: Render the atlas to disk and eyeball it**

Run:
```bash
mkdir -p workspace/studies/units-atlas-01-readout-inventory/viz
.venv/bin/python -c "
from v2ecoli.visualizations.units_atlas import UnitsAtlasVisualization
from v2ecoli.core import build_core
v = UnitsAtlasVisualization(config={'title':'Units Atlas'}, core=build_core())
v.accumulate({})
open('workspace/studies/units-atlas-01-readout-inventory/viz/units_atlas.html','w').write(v.render())
print('wrote units_atlas.html')
"
```
Expected: `wrote units_atlas.html`; open it and confirm dimension sections (mass/concentration/rate/…) with units.

- [ ] **Step 4: Confirm a workflow viz shows schema-sourced units**

Render a workflow report (or the mass panel) and confirm the mass y-axis reads `Mass (fg)` sourced from the schema (change `quantity[float,fg]` → `quantity[float,pg]` in a scratch check would flip the label — do NOT commit that probe).

- [ ] **Step 5: Final commit (if any verification fixups)**

```bash
git add -A && git commit -m "test(units): verification fixups" || echo "nothing to commit"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** resolver (Tasks 1-4) ↔ spec Piece 1; base hook + registration (Tasks 5-6) ↔ Piece 2; retrofits (Tasks 7-8) realize "display on all axes"; atlas (Tasks 9-11) ↔ Piece 3.
- **"Live from schema":** the index reads declared port types at build time (memoized), not a persisted sidecar — honors the spec's mechanism choice.
- **Known limitation (document, don't fix here):** `build_units_index` enumerates process classes with empty config; config-dependent processes are skipped, so a few paths may lack units. Widen `_iter_process_classes` later if a needed readout is missing. Path→store mapping assumes port path == store path (true for `listeners.*` by convention); wiring-remap is out of scope for v1.
- **Cross-repo:** Task 5 commits in `pbg-superpowers`; Task 6 editable-installs it into v2ecoli's venv. Both repos get separate PRs.
