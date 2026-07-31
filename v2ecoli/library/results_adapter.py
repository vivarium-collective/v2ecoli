"""Turn an emitter's ``results`` handle into what flush entities consume.

An ``EmitterResults`` handle resolves — via the emitter's own ``query()`` — to
whatever that emitter natively holds. For ``XArrayEmitter`` that is an
``xarray.DataTree`` read back off the written zarr store.

The existing v2ecoli visualizations want something much simpler: ``history``,
a flat ``list[dict]`` of per-tick rows (``{generation, time, dry_mass, …}``),
plus a ``metadata`` dict. This module is the seam between the two, so a
visualization that already works keeps working, now fed from a run's results
rather than from a parquet re-read.

The DataTree an ``XArrayEmitter`` writes is laid out per lineage::

    /experiment_id=…/variant=…/lineage_seed=…/          time_gen=1, time_gen=2, …
    /experiment_id=…/variant=…/lineage_seed=…/dry_mass/ generation=1, generation=2, …
    /experiment_id=…/variant=…/lineage_seed=…/…/        generation=…

so a variable's values live in a child group named after the variable, keyed by
generation, and the time axis lives beside them on the parent.
"""

from __future__ import annotations

from typing import Any

_TIME_PREFIX = "time_gen="
_GEN_PREFIX = "generation="


def _run_group(tree: Any) -> Any:
    """The node holding the per-generation time axes.

    That is the lineage group: everything for one run hangs off it.
    """
    for node in tree.subtree:
        if any(str(name).startswith(_TIME_PREFIX) for name in node.ds.data_vars):
            return node
    raise ValueError(
        "results DataTree has no time axis — expected a group with "
        f"{_TIME_PREFIX!r} variables, found groups: "
        f"{[n.path for n in tree.subtree]}")


def datatree_to_history(tree: Any) -> list[dict]:
    """Flatten an XArrayEmitter DataTree into per-tick rows.

    One row per emitted tick, carrying ``generation``, ``time`` and every
    recorded variable as a flat key — the shape
    ``v2ecoli.visualizations`` already reads.
    """
    run = _run_group(tree)

    times = {
        int(str(name)[len(_TIME_PREFIX):]): array
        for name, array in run.ds.data_vars.items()
        if str(name).startswith(_TIME_PREFIX)}

    history: list[dict] = []
    for generation in sorted(times):
        time_values = times[generation].values
        rows = [
            {"generation": generation, "time": float(time_values[index])}
            for index in range(len(time_values))]

        for child in run.children.values():
            variable = str(child.name)
            series = child.ds.data_vars.get(f"{_GEN_PREFIX}{generation}")
            if series is None:
                continue
            values = series.values
            for index in range(min(len(rows), len(values))):
                rows[index][variable] = float(values[index])

        history.extend(rows)

    return history


def datatree_metadata(tree: Any) -> dict:
    """Run metadata carried on the store, plus what the rows imply."""
    metadata = {
        str(key): value for key, value in (tree.attrs or {}).items()}
    try:
        run = _run_group(tree)
        metadata.update({
            str(key): value for key, value in (run.ds.attrs or {}).items()})
        metadata["n_generations"] = sum(
            1 for name in run.ds.data_vars
            if str(name).startswith(_TIME_PREFIX))
    except ValueError:
        pass
    return metadata


def results_to_visualization_inputs(handle: Any) -> dict:
    """Resolve a ``results`` handle into a visualization's input state.

    The handle is a reference — this is where the data is actually read.
    """
    tree = handle.resolve()
    return {
        "history": datatree_to_history(tree),
        "metadata": datatree_metadata(tree)}
