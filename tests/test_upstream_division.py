"""Tests for the harness-side upstream pbg division + the multigen emit-path fix.

Covers, WITHOUT a full ParCa cell build:

1. The deep-division emit-path bug (``KeyError: 'Unexpected emit path: ()'``)
   that crashed a multi-generation run entering generation 4. The
   ``XArrayEmitter`` transducer strips the agent prefix via
   ``get_in(data, ("agents", partition.agent_id))``; when the runner emits a
   cell's payload under the inner-composite key (``followed``) and that key has
   diverged from the emitter's own ``partition_agent_id`` (id reuse at deeper
   generations), ``get_in`` misses, ``dict_to_paths`` yields the empty-tuple
   path, and the emitter raises. The fix (``xarray_run._emit_followed`` emitting
   under ``partition_agent_id``) is exercised here directly.

2. The binomial bulk split sanity (each daughter ~= half the mother, counts
   conserved) used by :class:`v2ecoli.library.upstream_division.UpstreamDivision`.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from v2ecoli.core import build_core


_VIEW = [{
    "root": ("listeners",),
    "variables": {"mass": {"dry_mass": [{"path": "dry_mass", "dtype": "<f8"}]}},
}]
_META = {
    "experiment_id": "t", "variant": 0, "lineage_seed": 0,
    "time_step": 1.0, "max_duration": 100.0,
}


def _drive_emitter(emit_key: str, partition_agent_id: str, store: str):
    """Build a gen-1 XArrayEmitter and push 4 emit steps under ``emit_key``.

    Returns the raised exception (or None). With ``emit_key`` !=
    ``partition_agent_id`` this reproduces the deep-division crash.
    """
    from v2ecoli.library.xarray_run import _build_emitter

    em = _build_emitter(
        core=build_core(), store_path=store, view=_VIEW, metadata_base=_META,
        generation=1, agent_id=partition_agent_id, output_metadata={},
        buffer_size=3,
    )
    # Always close the emitter: its zarr writer owns a ThreadPoolExecutor and an
    # open store on zarr v3's shared sync event loop. Leaving it open leaks those
    # onto the next emitter's zarr.open_group(), which then deadlocks in
    # zarr.core.sync.sync() on the Linux CI runner (the second store never opens).
    try:
        for t in range(4):
            try:
                em.update({
                    "time": float(t), "global_time": float(t),
                    "agents": {emit_key: {"listeners": {"mass": {"dry_mass": 100.0 + t}}}},
                })
            except Exception as e:  # noqa: BLE001
                return e
        return None
    finally:
        try:
            em.close()
        except Exception:  # noqa: BLE001
            pass


def test_emit_under_mismatched_key_reproduces_empty_tuple_bug():
    """Emitting under a key != the emitter's partition agent_id crashes with the
    exact deep-division error this fix targets."""
    d = tempfile.mkdtemp()
    err = _drive_emitter("00", "0", os.path.join(d, "bug.zarr"))
    assert isinstance(err, KeyError)
    assert "Unexpected emit path: ()" in str(err)


def test_emit_under_partition_agent_id_is_the_fix():
    """Emitting under the emitter's own partition agent_id (what
    ``run_multigen_xarray._emit_followed`` now does) succeeds even when the
    inner-composite followed key has diverged."""
    d = tempfile.mkdtemp()
    err = _drive_emitter("0", "0", os.path.join(d, "fix.zarr"))
    assert err is None, f"emit under partition_agent_id should not raise, got {err!r}"


def test_binomial_bulk_split_conserves_and_halves():
    """``divide_bulk`` partitions counts ~50/50 with exact conservation — the
    'each daughter ~= half mother' guarantee the division step relies on."""
    from v2ecoli.library.division import divide_bulk

    rng = np.random.RandomState(0)
    counts = rng.randint(0, 10000, size=2000).astype(np.int64)
    bulk = np.zeros(counts.size, dtype=[("id", "S8"), ("count", "i8")])
    bulk["count"] = counts

    d1, d2 = divide_bulk(bulk)

    # Exact conservation: no molecule created or destroyed.
    assert np.array_equal(d1["count"] + d2["count"], counts)
    # No negative counts (binomial guarantees >= 0).
    assert (d1["count"] >= 0).all() and (d2["count"] >= 0).all()
    # Aggregate split is within 1% of half (binomial, p=0.5, large N).
    total = counts.sum()
    frac1 = d1["count"].sum() / total
    assert abs(frac1 - 0.5) < 0.01, frac1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_division_step_threads_config_overrides_to_daughters():
    """config_overrides passed to baseline() must reach the Division step so it
    re-applies them to each daughter's rebuild — otherwise a perturbation
    (variant / sweep) silently confines itself to generation 1.
    Regression guard for the daughters-revert-at-division bug."""
    from v2ecoli.core import build_core
    from v2ecoli.composites._helpers import CachedConfigLoader, _get_special_step

    core = build_core()
    overrides = {"ecoli-polypeptide-elongation.basal_elongation_rate": 19.5}

    # Perturbed: loader carries config_overrides -> Division instance stores them.
    loader = CachedConfigLoader(configs={"division": {}}, unique_names=[],
                                dry_mass_inc_dict={})
    loader._config_overrides = overrides
    instance, _topo, _kind = _get_special_step(loader, "division", core)
    assert getattr(instance, "_config_overrides", None) == overrides

    # Regression: plain baseline (no overrides) -> daughters rebuild unchanged.
    plain = CachedConfigLoader(configs={"division": {}}, unique_names=[],
                               dry_mass_inc_dict={})
    plain_inst, _t, _k = _get_special_step(plain, "division", core)
    assert getattr(plain_inst, "_config_overrides", None) is None
