"""Shrink a resolved composite state for VIEWING.

Extracted from the retired ``regenerate_viewers.py`` (the standalone composite
viewers hub, replaced by the read-only dashboard's Composites page). The only
piece still needed is ``trim_state_for_view`` — used by
``regenerate_composite_states.py`` when it pre-resolves the dashboard's
loom "Explore" pop-out states.
"""
from __future__ import annotations


def trim_state_for_view(obj, *, max_list: int = 8):
    """Shrink a resolved composite state for VIEWING.

    The bigraph STRUCTURE the viewers draw (stores, processes, wiring, schemas,
    describe() docs) lives in dicts and short path lists. The bulk weight is
    long molecule-data arrays — bulk counts (~16k), unique-molecule instance
    lists (active_ribosome ~12k, promoter ~5k, …) — which loom never renders.
    Capping every long list (numpy arrays included) keeps the cached state small
    (5.5 MB -> a few hundred KB) without touching structure.
    """
    if isinstance(obj, dict):
        return {k: trim_state_for_view(v, max_list=max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # NOTE: this caps EVERY list, including any structural list (e.g. a
        # process with >max_list ports expressed as a list). All current
        # composites are safe — their structural lists are short wiring paths
        # (<=4) — but a future composite with a long list in schema position
        # would be silently truncated; bump max_list or special-case if so.
        return [trim_state_for_view(v, max_list=max_list) for v in obj[:max_list]]
    # numpy (and other 1-D array-likes): cap then recurse on the python list.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist) and getattr(obj, "ndim", 0) >= 1:
        # Structured arrays (bulk / unique molecules) keep their dtype so the JSON
        # serializer (_json_default) can label rows by field name / id instead of
        # positional index. Just cap the row count; don't flatten to a list.
        if getattr(getattr(obj, "dtype", None), "names", None):
            return obj[:max_list]
        try:
            return trim_state_for_view(obj[:max_list].tolist(), max_list=max_list)
        except Exception:  # noqa: BLE001 — fall through to leave the value as-is
            pass
    return obj
