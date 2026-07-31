"""ColonyGrowthGif — animated GIF of a running colony (new-style Visualization).

New-style ``accumulate``/``render`` contract (see
``viva_superpowers.visualization.Visualization``): each tick, ``accumulate``
snapshots the cheap per-cell scalars (``location``, ``angle``, ``length``,
``radius``, ``mass``) from the colony's ``cells`` store plus the current
``global_time`` — never the deep embedded whole-cell state (see
``v2ecoli/colony.py``'s note on the "colony RAM leak": an emitter that
deep-copies the full ``cells`` map every tick costs ~1.6 MB/tick of RSS the
OS never reclaims; snapshotting five scalars per cell is cheap by
comparison). At the end of the run, ``render`` assembles the accumulated
frames into a ``trajectory`` and hands it to
``v2ecoli.colony_bench.viz.render_device_gif`` (viva-munk's
``simulation_to_gif`` under the hood), then inlines the resulting GIF as a
base64 ``<img>`` data URI.

WIRING RISK: this Step's typed input ports (``cells``, ``global_time``) are
only populated if a composite document explicitly wires them, e.g.::

    document['colony_growth_gif'] = {
        '_type': 'step',
        'address': 'local:ColonyGrowthGif',
        'config': {'env_size': env_size},
        'inputs': {'cells': ['cells'], 'global_time': ['global_time']},
    }

process-bigraph does NOT auto-wire a Step's input ports to same-named
top-level stores just because the names match (see ``v2ecoli/colony.py``:
every process/step in ``make_colony_document`` — ``ecoli``, ``multibody``,
``phenotype_recorder`` — carries an explicit ``inputs: {...}`` map). Without
that explicit wiring in whatever document instantiates this Step,
``accumulate`` is never called and ``render`` degrades to the "no frames"
note — the same failure mode that made the legacy ``ColonyVisualization``
render "?" (its ``history`` input was never supplied either).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from viva_superpowers.visualization import Visualization


class ColonyGrowthGif(Visualization):
    """Animated GIF of colony growth, built from per-tick cell snapshots.

    Inputs
    ------
    cells:       map[node] — the colony's ``cells`` store (agent_id -> body
                 dict with ``location``, ``angle``, ``length``, ``radius``,
                 ``mass``; see ``v2ecoli/colony.py``'s ``build_microbe``).
    global_time: float — the colony's simulated clock.

    Config
    ------
    env_size: float, default 30 — environment side length (µm), passed
              through to ``render_device_gif``.
    """

    config_schema = {
        **Visualization.config_schema,
        "env_size": {"_type": "float", "_default": 30.0},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "cells": "map[node]",
            "global_time": "float",
        }

    def accumulate(self, state: dict) -> None:
        """Snapshot cheap per-cell scalars — never the deep embedded state."""
        frames = getattr(self, "_frames", None)
        if frames is None:
            frames = self._frames = []
        cells = (state or {}).get("cells") or {}
        t = float((state or {}).get("global_time") or 0.0)
        snapshot: dict[str, dict] = {}
        for cid, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            loc = cell.get("location")
            if loc is None:
                continue
            snapshot[cid] = {
                "location": tuple(loc),
                "angle": float(cell.get("angle") or 0.0),
                "length": float(cell.get("length") or 2.0),
                "radius": float(cell.get("radius") or 0.5),
                "mass": float(cell.get("mass") or 0.0),
            }
        if not snapshot:
            return
        frames.append({"cells": snapshot, "time": t})

    def render(self) -> str:
        frames = getattr(self, "_frames", None) or []
        if not frames:
            return (
                "<div style='padding:12px;color:#64748b'>Colony-growth GIF "
                "unavailable: no cell frames were accumulated (check that "
                "this Step's 'cells'/'global_time' inputs are wired to the "
                "colony's stores).</div>"
            )

        cfg = getattr(self, "config", None) or {}
        env_size = float(cfg.get("env_size", 30.0) or 30.0)

        try:
            from v2ecoli.colony_bench.viz import render_device_gif

            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "colony_growth.gif"
                render_device_gif(frames, out_path, env_size=env_size)
                gif_bytes = out_path.read_bytes()
        except Exception as e:  # noqa: BLE001 — never raise, degrade instead
            return (
                "<div style='padding:12px;color:#64748b'>Colony-growth GIF "
                f"unavailable: {e}</div>"
            )

        b64 = base64.b64encode(gif_bytes).decode("ascii")
        return f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>'
