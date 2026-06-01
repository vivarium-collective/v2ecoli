"""Render Visualization Steps from a study's parquet-runs output.

The vivarium-dashboard currently reads sim data only from a SQLite emitter's
``history`` table; its inputs_map → port resolution speaks the SQLite shape.
When a study runs with ``parquet_emitter()`` instead of ``sqlite_emitter()``,
the dashboard's data-resolution layer has nothing to read.

This module is the parquet equivalent. Given a study slug, it:

  1. Locates the latest run under ``workspace/studies/<slug>/parquet-runs/``
  2. Loads all ``history/<run_id>/.../history/*.pq`` files into a single
     time-sorted Polars DataFrame
  3. For each ``visualizations[]`` entry in ``study.yaml``, resolves
     ``inputs_map`` (dotted-path observables → parquet columns, with
     ``.`` translated to ``__``) into a port-keyed state dict
  4. Instantiates the Visualization subclass (resolved from ``address:
     local:<ClassName>`` against the workspace's discovered Visualization
     subclasses) and calls ``update(state)`` to get rendered HTML
  5. Writes ``workspace/studies/<slug>/viz/<viz_name>.html``

Multi-source vizzes (``config.sources: [sim_name_a, sim_name_b]``) are not
yet supported — the current implementation renders the latest run only.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _find_workspace_root(start: Path | None = None) -> Path:
    """Walk up to find workspace.yaml."""
    cur = (start or Path.cwd()).resolve()
    while cur != cur.parent:
        if (cur / "workspace.yaml").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        "no workspace.yaml found walking up from "
        f"{(start or Path.cwd()).resolve()}"
    )


def find_latest_parquet_run(study_slug: str,
                            workspace_root: Path | None = None) -> Path:
    """Return the most-recent ``workspace/studies/<slug>/parquet-runs/<run_id>/``.

    Picks by directory mtime. Raises FileNotFoundError if no runs exist.
    """
    ws = workspace_root or _find_workspace_root()
    parquet_root = ws / "workspace" / "studies" / study_slug / "parquet-runs"
    if not parquet_root.exists():
        raise FileNotFoundError(
            f"no parquet-runs directory for study {study_slug!r} at {parquet_root}"
        )
    runs = [p for p in parquet_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(
            f"no run directories under {parquet_root}"
        )
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def load_run_history(parquet_run_dir: Path) -> pl.DataFrame:
    """Load all history/*.pq for one run into a single time-sorted DataFrame.

    Files are named by the cumulative step number at flush time
    (``400.pq``, ``800.pq``, ``1801.pq``, ...). Sort by step in the filename
    so concatenation preserves time order even when the filesystem returns
    them in arbitrary order.
    """
    history_root = parquet_run_dir / "history"
    if not history_root.exists():
        raise FileNotFoundError(
            f"no history/ directory under {parquet_run_dir}"
        )
    pq_files = sorted(history_root.rglob("*.pq"),
                      key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not pq_files:
        raise FileNotFoundError(
            f"no .pq history files under {history_root}"
        )
    frames = [pl.read_parquet(p) for p in pq_files]
    # ``diagonal_relaxed`` (not ``vertical_relaxed``) is required because the
    # parquet history can have schema drift across batches — a listener may
    # come and go as the cell-state shape changes tick to tick. ``diagonal``
    # fills missing columns with nulls; ``_relaxed`` also coerces dtypes.
    df = pl.concat(frames, how="diagonal_relaxed")
    if "global_time" in df.columns:
        df = df.sort("global_time")
    return df


def resolve_inputs_map(df: pl.DataFrame,
                       inputs_map: dict[str, str]) -> dict[str, list[Any]]:
    """Translate inputs_map dotted-paths to parquet columns + return values.

    Parquet columns flatten the bigraph dotted path with ``__`` (vEcoli
    convention; see USE_UINT16 / VECOLI_PARQUET_DTYPE_OVERRIDES). The
    inputs_map uses ``.`` — translate, look up, return as a Python list.
    Missing columns become ``None`` so partial-data vizzes still render
    (they tolerate empty ports).
    """
    out: dict[str, list[Any]] = {}
    for port, dotted_path in (inputs_map or {}).items():
        col = dotted_path.replace(".", "__")
        if col in df.columns:
            out[port] = df[col].to_list()
        else:
            out[port] = None
    return out


_VIZ_CLASS_CACHE: dict[str, type] | None = None


def _discover_visualization_classes() -> dict[str, type]:
    """Walk v2ecoli.visualizations and collect Visualization subclasses by name.

    The vivarium-dashboard's ``local:<ClassName>`` address scheme picks the
    workspace package's Visualization classes by short name. Replicate that
    lookup here.
    """
    global _VIZ_CLASS_CACHE
    if _VIZ_CLASS_CACHE is not None:
        return _VIZ_CLASS_CACHE

    from pbg_superpowers.visualization import Visualization

    pkg = importlib.import_module("v2ecoli.visualizations")
    classes: dict[str, type] = {}
    for _finder, name, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"v2ecoli.visualizations.{name}")
        except Exception:
            continue
        for cls_name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj is Visualization or not issubclass(obj, Visualization):
                continue
            classes[cls_name] = obj
    _VIZ_CLASS_CACHE = classes
    return classes


def _resolve_address(address: str) -> type:
    """``local:DnaAStateVisualization`` → the class object."""
    if ":" not in address:
        raise ValueError(f"unrecognized address {address!r} (expected scheme:name)")
    scheme, name = address.split(":", 1)
    if scheme != "local":
        raise ValueError(
            f"address scheme {scheme!r} not supported (only 'local:' for now)"
        )
    classes = _discover_visualization_classes()
    if name not in classes:
        raise KeyError(
            f"no Visualization class named {name!r} in v2ecoli.visualizations. "
            f"Available: {sorted(classes)}"
        )
    return classes[name]


_CORE = None


def _get_core():
    """Lazily build a v2ecoli core for Visualization instantiation.

    Step.__init__ requires a real core with ``fill(config_schema, config)``
    so we go through v2ecoli.core.build_core. Cached because build_core is
    relatively expensive and viz instantiation is stateless.
    """
    global _CORE
    if _CORE is None:
        from v2ecoli.core import build_core
        _CORE = build_core()
    return _CORE


def _instantiate_visualization(viz_cls: type, config: dict) -> Any:
    """Build a Visualization instance with a real v2ecoli core."""
    return viz_cls(config=config or {}, core=_get_core())


def render_study_visualizations(study_slug: str,
                                workspace_root: Path | None = None,
                                out_dir: Path | None = None
                                ) -> list[Path]:
    """Render every viz in ``workspace/studies/<slug>/study.yaml`` to HTML files.

    Reads the latest parquet run for the study, resolves each viz's
    inputs_map, calls the Visualization's ``update(state)``, and writes
    ``<out_dir>/<viz_name>.html`` (default ``workspace/studies/<slug>/viz/``).

    Returns the list of written file paths.
    """
    ws = workspace_root or _find_workspace_root()
    study_yaml = ws / "workspace" / "studies" / study_slug / "study.yaml"
    if not study_yaml.exists():
        raise FileNotFoundError(study_yaml)

    spec = yaml.safe_load(study_yaml.read_text(encoding="utf-8")) or {}
    viz_specs = spec.get("visualizations") or []
    if not viz_specs:
        return []

    run_dir = find_latest_parquet_run(study_slug, workspace_root=ws)
    df = load_run_history(run_dir)

    out_dir = out_dir or (ws / "workspace" / "studies" / study_slug / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for vs in viz_specs:
        name = vs.get("name") or "viz"
        address = vs.get("address") or ""
        cfg = vs.get("config") or {}
        inputs_map = cfg.get("inputs_map") or {}

        try:
            viz_cls = _resolve_address(address)
        except (KeyError, ValueError) as e:
            print(f"  ✗ {name}: {e}")
            continue

        state = resolve_inputs_map(df, inputs_map)

        viz = _instantiate_visualization(viz_cls, cfg)
        result = viz.update(state)
        html = (result or {}).get("html") or ""
        if not html:
            # New-style Visualization that uses accumulate/render rather
            # than update — call render() after a single accumulate(state).
            viz.accumulate(state)
            html = viz.render()

        out_path = out_dir / f"{name}.html"
        out_path.write_text(html, encoding="utf-8")
        written.append(out_path)
        print(f"  ok {name}: {out_path.relative_to(ws)} ({len(html)} chars)")
    return written
