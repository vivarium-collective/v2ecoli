"""Run viva-munk microfluidic device documents with simple agents.

Wraps viva-munk's ``mother_machine_document`` and ``daughter_machine_document``
(narrow dead-end channels + flow; a chamber with an absorbing wall) — the
canonical devices, populated by viva-munk's own grow/divide simple agents — and
samples a geometry+phenotype trajectory each step. The viva-munk viz Steps are
stripped so we drive the animation/figures ourselves (viz.py).

The sampled trajectory frame shape matches what ``phenotype_extractor`` consumes
(``{time, cells: {id: {mass, length, ...}}}``) and additionally carries
``location``/``angle``/``radius`` for the GIF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from process_bigraph import Composite  # noqa: E402
from viva_munk import core_import  # noqa: E402
from viva_munk.experiments.documents.mother_machine import mother_machine_document  # noqa: E402
from viva_munk.experiments.documents.daughter_machine import daughter_machine_document  # noqa: E402

_DEVICES = {
    "mother_machine": mother_machine_document,
    "daughter_machine": daughter_machine_document,
}
_VIZ_KEYS = ("stores", "multibody_viz", "cell_mass_traces")


def build_device(kind: str, config: dict | None = None):
    """Build a device Composite (viz Steps stripped) + geometry meta."""
    if kind not in _DEVICES:
        raise ValueError(f"unknown device: {kind!r} (have {sorted(_DEVICES)})")
    core = core_import()
    doc = _DEVICES[kind](config or {})
    mb = doc["multibody"]["config"]
    meta = {"env_size": float(mb["env_size"]), "barriers": mb.get("barriers", [])}
    for k in _VIZ_KEYS:
        doc.pop(k, None)
    comp = Composite({"state": doc}, core=core)
    return comp, meta


def _sample(state, dt: float, n_prev: int) -> dict:
    cells: dict[str, dict] = {}
    for cid, c in state.get("cells", {}).items():
        loc = c.get("location")
        cells[cid] = {
            "mass": float(c.get("mass") or 0.0),
            "length": float(c.get("length") or 0.0),
            "location": tuple(loc) if loc is not None else None,
            "angle": float(c.get("angle") or 0.0),
            "radius": float(c.get("radius") or 0.5),
        }
    return {"time": float(state.get("global_time", n_prev * dt)), "cells": cells}


def run_device(kind: str, *, n_steps: int, dt: float = 30.0,
               config: dict | None = None) -> dict[str, Any]:
    """Run a device for ``n_steps`` of ``dt`` seconds, sampling each step.

    Returns ``{trajectory, phenotypes, n_final, meta}`` where ``meta`` carries
    ``env_size`` + ``barriers`` for the GIF and ``phenotypes`` is the
    ``phenotype_extractor`` result over the sampled trajectory.
    """
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor

    comp, meta = build_device(kind, config)
    trajectory = [_sample(comp.state, dt, 0)]
    for _ in range(n_steps):
        comp.run(dt)
        trajectory.append(_sample(comp.state, dt, len(trajectory)))
    return {
        "trajectory": trajectory,
        "phenotypes": phenotype_extractor(trajectory),
        "n_final": len(comp.state.get("cells", {})),
        "meta": meta,
    }


def _flow_regions(kind: str, config: dict | None, meta: dict) -> list[dict]:
    """The wash-out boundary to shade in the GIF (cells removed past it)."""
    config = config or {}
    if kind == "mother_machine":
        y = float(config.get("channel_height", 20.0))
        return [{"y_min": y, "y_max": meta["env_size"]}]
    if kind == "daughter_machine":
        env = float(config.get("env_size", meta["env_size"]))
        x = float(config.get("flow_x", env * 0.85))
        return [{"x_min": x, "x_max": meta["env_size"]}]
    return []


def run_device_study(kind: str, out_dir: str | Path, *, n_steps: int, dt: float = 30.0,
                     config: dict | None = None, label: str | None = None,
                     gif_frames: int = 80) -> dict[str, Any]:
    """Run a device and write the study's artifacts under ``out_dir``.

    Produces ``charts/colony.gif`` (running animation), the three phenotype
    distribution figures under ``charts/``, and ``phenotypes.json`` +
    ``summary.json`` at ``out_dir`` root. Returns the ``run_device`` result.
    """
    from v2ecoli.colony_bench import viz

    out_dir = Path(out_dir)
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    label = label or kind.replace("_", " ")

    out = run_device(kind, n_steps=n_steps, dt=dt, config=config)
    meta = out["meta"]
    gif_skip = max(1, n_steps // max(1, gif_frames))

    gif_path = charts / "colony.gif"
    viz.render_device_gif(
        out["trajectory"], gif_path,
        env_size=meta["env_size"], barriers=meta["barriers"],
        flow_regions=_flow_regions(kind, config, meta), skip_frames=gif_skip,
    )
    gif_path.with_suffix(".meta.json").write_text(json.dumps({
        "title": f"{label}: running animation",
        "caption": "Cells as capsules (lineage-coloured); shaded band = wash-out boundary.",
    }, indent=2))

    viz.render_phenotype_figures(out["phenotypes"], charts, label=label)

    (out_dir / "phenotypes.json").write_text(json.dumps(out["phenotypes"], indent=2))
    ph = out["phenotypes"]
    (out_dir / "summary.json").write_text(json.dumps({
        "device": kind, "n_steps": n_steps, "dt": dt, "n_final": out["n_final"],
        "n_division_events": ph["n_division_events"],
        "mean_length_at_division": (
            float(sum(ph["size_at_division"]["length"]) / len(ph["size_at_division"]["length"]))
            if ph["size_at_division"]["length"] else None),
    }, indent=2))
    return out

