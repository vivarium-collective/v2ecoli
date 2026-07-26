"""Bounded-RAM per-cell phenotype recorder for the colony composite.

Why this exists
---------------
The colony's default outer emitter wired the whole ``cells`` map into a
``RAMEmitter``. Measured cost: ``emit_cells=True`` grows RSS ~1.5 MB/tick vs
~0.09 MB/tick with ``emit_cells=False`` — i.e. **the outer emitter is ~94% of
the "colony RAM leak"** the investigation chased as a mysterious native/C leak.

Isolation runs (colonies) showed the cost is **not** storage: a no-op emitter
and a typed-shallow emit schema *both* still leaked ~1.6 MB/tick. The leak is
the process-bigraph **emit-gather deep-copying the heavy cells map** (each
cell's embedded 55-process WCM state) every tick to hand it to an *emitter*;
the freed copies aren't returned to the OS, so RSS climbs (and is invisible to
tracemalloc because the bulk is numpy buffers).

Crucially, a **process** wired to ``cells`` gets a cheap typed *view*, not a
deep copy — that's why ``PymunkProcess`` (also wired to ``['cells']``) is
leak-free. So per-cell capture must run as a PROCESS, not an emitter.

``ColonyPhenotypeRecorder`` is a process typed ``map[pymunk_agent]`` (the same
shallow view PymunkProcess uses, which never materialises each cell's ``ecoli``
sub-state). Each tick it reads the scalar phenotype panel per live cell and
appends it to resizable zarr arrays in tidy/long format (one row per
``(tick, cell)``), keeping only a bounded flush buffer in RAM. The result is an
xarray-readable zarr store with exactly the fields the phenotype studies need
(size-at-division, inter-division time, added size, growth rate) at O(1) memory.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import math
import os
import platform
from typing import Any, Dict

from process_bigraph import Process


def release_memory() -> None:
    """Return freed allocator pages to the OS (bounds long-run colony RSS).

    The colony's dominant per-tick RAM growth is allocator ARENA RETENTION: the
    inner WCM's polypeptide-elongation step allocates large numpy working arrays
    (~1.3 MB/tick via buildSequences/polymerize) and scipy's LSODA integrator
    allocates rwork/iwork every solve; the pages are freed but the allocator
    doesn't hand them back, so RSS climbs (invisible to tracemalloc). On glibc
    (Linux — the HPC target) ``malloc_trim(0)`` reclaims them. macOS has no
    equivalent that helps here, so this is effectively a no-op on the dev mini
    and must be validated on Linux. Cheap enough to call every flush.
    """
    gc.collect()
    try:
        if platform.system() == "Linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Columns written to zarr per cell per tick. ``x``/``y`` come from ``location``;
# ``volume`` is computed from the capsule geometry (length + radius); the rest
# are read straight off the pymunk_agent view.
COLUMNS = ["mass", "length", "volume", "x", "y", "angle"]


def _num(v: Any) -> float:
    """Coerce a possibly-unit / possibly-None value to a plain float (NaN on miss)."""
    if v is None:
        return math.nan
    mag = getattr(v, "magnitude", None)  # pint / unum quantities
    if mag is not None:
        v = mag
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _capsule_volume(length: float, radius: float) -> float:
    """Volume of a rod-shaped cell = cylinder + 2 hemispherical caps."""
    if math.isnan(length) or math.isnan(radius):
        return math.nan
    return math.pi * radius * radius * length + (4.0 / 3.0) * math.pi * radius ** 3


class ColonyPhenotypeRecorder(Process):
    """Stream a per-cell phenotype panel to a zarr store with bounded RAM.

    Wire it like PymunkProcess (a process, so it gets a leak-free typed view of
    the cells), pointed at a store path::

        {
          '_type': 'process',
          'address': 'local:ColonyPhenotypeRecorder',
          'config': {'out_uri': '.../runs.<id>.zarr'},
          'interval': 1.0,
          'inputs': {'cells': ['cells'], 'global_time': ['global_time']},
        }

    The store holds one group of 1-D arrays along a ``record`` dimension:
    ``time, cell_id, mass, length, volume, x, y, angle`` — tidy/long format,
    loaded with ``xarray.open_zarr`` and pivoted per cell.
    """

    config_schema = {
        "out_uri": {"_type": "string", "_default": ""},
        "flush_every": {"_type": "integer", "_default": 50},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.out_uri = (config or {}).get("out_uri") or ""
        self.flush_every = int((config or {}).get("flush_every") or 50)
        self._cols = {"time": [], "cell_id": []}
        for c in COLUMNS:
            self._cols[c] = []
        self._ticks_buffered = 0
        self._zroot = None
        self._string_dtype = None

    # A process typed map[pymunk_agent] -> engine hands us the shallow per-cell
    # view (mass/length/radius/angle/location), never the deep `ecoli` state.
    def inputs(self) -> Dict:
        return {"cells": "map[pymunk_agent]", "global_time": "float"}

    # Pure side-effect recorder: it writes to zarr, produces no state updates.
    def outputs(self) -> Dict:
        return {}

    def update(self, state, interval=None) -> Dict:
        t = _num((state or {}).get("global_time", 0.0))
        cells = (state or {}).get("cells") or {}
        for cid, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            mass = _num(cell.get("mass"))
            length = _num(cell.get("length"))
            radius = _num(cell.get("radius"))
            angle = _num(cell.get("angle"))
            loc = cell.get("location") or ()
            x = _num(loc[0]) if len(loc) > 0 else math.nan
            y = _num(loc[1]) if len(loc) > 1 else math.nan
            self._cols["time"].append(t)
            self._cols["cell_id"].append(str(cid))
            self._cols["mass"].append(mass)
            self._cols["length"].append(length)
            self._cols["volume"].append(_capsule_volume(length, radius))
            self._cols["x"].append(x)
            self._cols["y"].append(y)
            self._cols["angle"].append(angle)
        self._ticks_buffered += 1
        if self._ticks_buffered >= self.flush_every:
            self.flush()
        return {}

    # ---- zarr sink ---------------------------------------------------------
    def _ensure_store(self):
        import numpy as np
        import zarr

        if self._zroot is not None or not self.out_uri:
            return
        os.makedirs(os.path.dirname(self.out_uri) or ".", exist_ok=True)
        self._string_dtype = (
            np.dtypes.StringDType() if hasattr(np, "dtypes") else object)
        root = zarr.open_group(self.out_uri, mode="a")
        specs = [("time", "<f8"), ("cell_id", self._string_dtype)]
        specs += [(c, "<f4") for c in COLUMNS]
        for name, dt in specs:
            if name in root:
                continue
            arr = root.create_array(
                name, shape=(0,), chunks=(4096,), dtype=dt,
                dimension_names=["record"],  # zarr v3 xarray-readability
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["record"]  # v2 fallback
        self._zroot = root

    def flush(self) -> None:
        n = len(self._cols["time"])
        self._ticks_buffered = 0
        if not n:
            return
        import numpy as np

        self._ensure_store()
        if self._zroot is None:  # no store configured -> drop (still bounded)
            for k in self._cols:
                self._cols[k].clear()
            return
        for name in ["time", "cell_id"] + COLUMNS:
            arr = self._zroot[name]
            old = arr.shape[0]
            arr.resize((old + n,))
            if name == "cell_id":
                arr[old:old + n] = np.asarray(self._cols[name], dtype=arr.dtype)
            else:
                arr[old:old + n] = np.asarray(self._cols[name], dtype="<f8")
            self._cols[name].clear()
        # Reclaim the WCM's per-tick allocator churn on the HPC (Linux) target,
        # amortised at the flush cadence rather than every tick.
        release_memory()

    # Called by the run harness at end-of-run to persist the tail buffer.
    def close(self, *args, **kwargs) -> None:
        self.flush()
