"""Decide whether a lineage cell_mass trajectory is physically valid.

A valid whole-cell basal run grows cell_mass ~2x within a generation and then
halves at division. The pre-fix wrapper bug made cell_mass explode ~18x in one
generation (fail-open partition gate -> evolvers re-applied bulk deltas every
tick). This module turns that distinction into a hard PASS/FAIL gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr


def segment_generations(cell_mass: np.ndarray, *, drop_frac: float = 0.6) -> list[tuple[int, int]]:
    cm = np.asarray(cell_mass, dtype=float)
    if cm.size == 0:
        return []
    bounds = [0]
    for i in range(cm.size - 1):
        if cm[i] > 0 and cm[i + 1] < drop_frac * cm[i]:
            bounds.append(i + 1)
    bounds.append(cm.size)
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


@dataclass
class Verdict:
    physical: bool
    generations_reached: int
    divisions_detected: int
    per_gen_ratios: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def assess_physical(
    cell_mass: np.ndarray,
    *,
    min_generations: int = 2,
    doubling_band: tuple[float, float] = (1.5, 3.5),
) -> Verdict:
    cm = np.asarray(cell_mass, dtype=float)
    segs = segment_generations(cm)
    divisions = max(len(segs) - 1, 0)
    # complete generations are all but the last (the last may be mid-cycle, no division yet)
    complete = segs[:-1] if len(segs) >= 1 else []
    ratios: list[float] = []
    reasons: list[str] = []
    lo, hi = doubling_band
    for (s, e) in complete:
        if e - s < 2 or cm[s] <= 0:
            continue
        r = float(cm[e - 1] / cm[s])
        ratios.append(r)
        if not (lo <= r <= hi):
            reasons.append(f"generation [{s}:{e}] growth ratio {r:.2f} outside physical band {doubling_band}")
    if divisions < min_generations - 1:
        reasons.append(
            f"only {divisions} division(s) detected; require >= {min_generations - 1} "
            f"for {min_generations} generations"
        )
    physical = len(reasons) == 0 and len(ratios) >= max(min_generations - 1, 1)
    if len(ratios) < max(min_generations - 1, 1) and not reasons:
        reasons.append("no complete generation with a measurable growth ratio")
        physical = False
    return Verdict(
        physical=physical,
        generations_reached=len(segs),
        divisions_detected=divisions,
        per_gen_ratios=ratios,
        reasons=reasons,
    )


def _zarr_find_cell_mass(store_path: str) -> np.ndarray:
    """Walk the zarr group hierarchy to find cell_mass arrays.

    run_multigen_xarray writes stores with the Hive-style layout:
        {store}/experiment_id=.../variant=.../lineage_seed=.../cell_mass/generation=N

    xr.open_zarr at the root sees no variables (only groups), so we walk zarr
    directly: find every array whose path ends in .../cell_mass/generation=<int>,
    sort by generation index, concatenate in order.
    """
    import zarr  # local import — zarr is only needed in this fallback path

    root = zarr.open(store_path)

    def _iter_arrays(grp, path=""):
        for name, child in grp.members():
            child_path = f"{path}/{name}" if path else name
            if hasattr(child, "members"):
                yield from _iter_arrays(child, child_path)
            else:
                # child is a zarr Array
                yield child_path, child

    # Collect all arrays whose leaf parent is "cell_mass" (path ends in
    # .../cell_mass/generation=N).
    gen_arrays: list[tuple[int, np.ndarray]] = []
    for arr_path, arr in _iter_arrays(root):
        parts = arr_path.split("/")
        if len(parts) >= 2 and parts[-2] == "cell_mass" and parts[-1].startswith("generation="):
            try:
                gen_idx = int(parts[-1].split("=", 1)[1])
            except ValueError:
                continue
            gen_arrays.append((gen_idx, np.asarray(arr[:], dtype=float)))

    if not gen_arrays:
        raise ValueError(
            f"no 'cell_mass/generation=*' arrays found in {store_path} "
            f"(zarr walk found no matching paths)"
        )

    gen_arrays.sort(key=lambda x: x[0])
    return np.concatenate([a for _, a in gen_arrays]).reshape(-1)


def load_cell_mass(store_path: str) -> np.ndarray:
    """Return the time-ordered cell_mass series from a run_multigen_xarray store.

    Tries two strategies:

    1. Flat xarray: ``xr.open_zarr(store_path)`` and search ``data_vars`` for a
       variable whose leaf name is ``cell_mass`` (legacy / simple stores).
    2. Nested zarr groups: walk the zarr hierarchy to find arrays at the path
       ``…/cell_mass/generation=<int>`` emitted by the Hive-style
       ``run_multigen_xarray`` stores (``experiment_id=…/variant=…/lineage_seed=…/
       cell_mass/generation=N``). Concatenates all generations in order.

    Returns a 1-D time-ordered float array. Raises ``ValueError`` if no
    ``cell_mass`` data is found by either strategy.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(store_path)
    name = next((v for v in ds.data_vars if str(v).split("/")[-1] == "cell_mass"), None)
    if name is not None:
        return np.asarray(ds[name].values, dtype=float).reshape(-1)

    # Flat xarray found nothing — fall back to zarr group walk.
    return _zarr_find_cell_mass(store_path)
