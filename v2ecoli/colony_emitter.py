"""Bounded-RAM per-cell phenotype emitter for the colony composite.

Why this exists
---------------
The colony's default outer emitter wired the whole ``cells`` map into a
``RAMEmitter``, which appended every cell's (numpy-array-heavy) state to an
in-memory history *every tick*. Measured cost: ``emit_cells=True`` grows RSS
~1.5 MB/tick vs ~0.09 MB/tick with ``emit_cells=False`` — i.e. **the outer
emitter is ~94% of the "colony RAM leak"** the colonies investigation chased
as a mysterious native/C leak. It only looked native because the accumulated
state is numpy buffers, which ``tracemalloc`` cannot see.

The stock ``pbg_emitters`` ``XArrayEmitter`` is the right *idea* (stream to a
zarr store instead of RAM) but its "colony" strategy assumes vEcoli
lineage-string agent ids (``generation == len(agent_id)``, ``parent == id[:-1]``)
and one partition per lineage. v2ecoli colonies use arbitrary ids (``a_0``,
daughters ``a_0_0``) and a single dynamic cells map, which that layout rejects.

``ColonyPhenotypeEmitter`` is the minimal, colony-native fix: each tick it
extracts a small scalar **phenotype panel** per live cell and appends it to
resizable zarr arrays in tidy/long format (one row per ``(tick, cell)``),
keeping only a bounded flush buffer in RAM. The result is an xarray-readable
zarr store carrying exactly the fields Part B needs (size-at-division,
inter-division time, added size, growth rate) at O(1) memory.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

from process_bigraph.emitter import Emitter

# Fields emitted per cell per tick. ``x``/``y`` are unpacked from ``location``;
# all others are read straight off the cell body.
DEFAULT_FIELDS = ["mass", "length", "volume", "x", "y", "angle"]


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


class ColonyPhenotypeEmitter(Emitter):
    """Stream a per-cell phenotype panel to a zarr store with bounded RAM.

    Wire it like the RAM emitter but pointed at a store path::

        {
          '_type': 'step',
          'address': 'local:ColonyPhenotypeEmitter',
          'config': {'emit': {'cells': 'map', 'global_time': 'float'},
                     'out_uri': '.../runs.<id>.zarr'},
          'inputs': {'cells': ['cells'], 'global_time': ['global_time']},
        }

    The store holds one group of 1-D arrays along a ``record`` dimension:
    ``time, cell_id, mass, length, volume, x, y, angle`` — tidy/long format,
    trivially loaded with ``xarray.open_zarr`` and pivoted per cell.
    """

    config_schema = {
        **Emitter.config_schema,  # {'emit': 'schema'}
        "out_uri": {"_type": "string", "_default": ""},
        "fields": {"_type": "list", "_default": DEFAULT_FIELDS},
        "flush_every": {"_type": "integer", "_default": 50},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.out_uri = (config or {}).get("out_uri") or ""
        self.fields = list((config or {}).get("fields") or DEFAULT_FIELDS)
        self.flush_every = int((config or {}).get("flush_every") or 50)
        # In-RAM flush buffer (bounded: cleared every ``flush_every`` ticks).
        self._cols = {"time": [], "cell_id": []}
        for f in self.fields:
            self._cols[f] = []
        self._ticks_buffered = 0
        self._zroot = None          # lazy-opened zarr group
        self._string_dtype = None

    # inputs() falls back to the base Emitter (returns config['emit']). We
    # rely on a TYPED shallow emit schema (declared in colony.py) so the engine
    # gathers only the per-cell scalar panel and never descends into each cell's
    # heavy `ecoli` sub-state — descending is what deep-copied ~1.6 MB/tick and
    # made the colony look like it had a native leak.

    def update(self, state) -> Dict:
        t = _num((state or {}).get("global_time", 0.0))
        cells = (state or {}).get("cells") or {}
        for cid, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            self._cols["time"].append(t)
            self._cols["cell_id"].append(str(cid))
            loc = cell.get("location") or ()
            for f in self.fields:
                if f == "x":
                    v = _num(loc[0]) if len(loc) > 0 else math.nan
                elif f == "y":
                    v = _num(loc[1]) if len(loc) > 1 else math.nan
                else:
                    v = _num(cell.get(f))
                self._cols[f].append(v)
        self._ticks_buffered += 1
        if self._ticks_buffered >= self.flush_every:
            self.flush()
        return {}

    # ---- zarr sink ---------------------------------------------------------
    def _ensure_store(self, n_new: int):
        import numpy as np
        import zarr

        if self._zroot is not None:
            return
        if not self.out_uri:
            return
        os.makedirs(os.path.dirname(self.out_uri) or ".", exist_ok=True)
        # Variable-length UTF-8 for cell ids; float32 for the numeric panel.
        self._string_dtype = np.dtypes.StringDType() if hasattr(np, "dtypes") else object
        root = zarr.open_group(self.out_uri, mode="a")
        specs = [("time", "<f8"), ("cell_id", None)] + [(f, "<f4") for f in self.fields]
        for name, dt in specs:
            if name in root:
                continue
            if name == "cell_id":
                arr = root.create_array(
                    name, shape=(0,), chunks=(4096,), dtype=self._string_dtype)
            else:
                arr = root.create_array(
                    name, shape=(0,), chunks=(4096,), dtype=dt)
            # xarray-readability: label the single dimension.
            arr.attrs["_ARRAY_DIMENSIONS"] = ["record"]
        self._zroot = root

    def flush(self) -> None:
        n = len(self._cols["time"])
        self._ticks_buffered = 0
        if not n:
            return
        if os.environ.get("COLONY_EMIT_NOOP"):  # isolate storage from wiring cost
            for k in self._cols:
                self._cols[k].clear()
            return
        import numpy as np

        self._ensure_store(n)
        if self._zroot is None:  # no store configured -> just drop (bounded)
            for k in self._cols:
                self._cols[k].clear()
            return
        for name, arr in (("time", None),):  # placeholder to keep import tidy
            pass
        for name in ["time", "cell_id"] + self.fields:
            arr = self._zroot[name]
            old = arr.shape[0]
            arr.resize((old + n,))
            if name == "cell_id":
                arr[old:old + n] = np.asarray(self._cols[name], dtype=arr.dtype)
            else:
                arr[old:old + n] = np.asarray(self._cols[name], dtype="<f8")
            self._cols[name].clear()

    # Composite/harness calls this at end-of-run to persist the tail buffer.
    def close(self, *args, **kwargs) -> None:
        self.flush()
