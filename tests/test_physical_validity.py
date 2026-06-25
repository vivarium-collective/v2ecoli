import numpy as np
import pytest
import xarray as xr
from scripts._compare.physical_validity import segment_generations, assess_physical, load_cell_mass


def _doubling_trajectory(n_gens=3, steps_per_gen=50, m0=5000.0):
    """Physical: linear growth m -> ~2m within a gen, then halve at division."""
    segs = []
    m = m0
    for _ in range(n_gens):
        seg = np.linspace(m, 2 * m, steps_per_gen, endpoint=False)
        segs.append(seg)
        m = seg[-1] / 2.0  # division halves the mother into the next founder
    return np.concatenate(segs)


def test_segment_splits_at_each_division():
    cm = _doubling_trajectory(n_gens=3, steps_per_gen=50)
    segs = segment_generations(cm)
    assert len(segs) == 3
    # half-open, contiguous, covering
    assert segs[0][0] == 0
    assert segs[-1][1] == len(cm)


def test_physical_doubling_passes():
    cm = _doubling_trajectory(n_gens=3)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is True
    assert v.divisions_detected == 2
    assert all(1.5 <= r <= 3.5 for r in v.per_gen_ratios)


def test_mass_explosion_fails():
    # 5k -> ~90k in one generation (the pre-fix bug), no division
    cm = np.linspace(5000.0, 90000.0, 80)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is False
    assert any("ratio" in r.lower() or "division" in r.lower() for r in v.reasons)


def test_truncated_run_fails_on_generation_count():
    # one clean generation then nothing — fewer divisions than required
    cm = _doubling_trajectory(n_gens=1)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is False
    assert any("division" in r.lower() or "generation" in r.lower() for r in v.reasons)


def test_load_cell_mass_from_zarr(tmp_path):
    store = str(tmp_path / "lineage.zarr")
    cm = np.linspace(5000.0, 10000.0, 30)
    ds = xr.Dataset({"cell_mass": ("time", cm)}, coords={"time": np.arange(30)})
    ds.to_zarr(store, mode="w", consolidated=False)
    out = load_cell_mass(store)
    assert out.shape == (30,)
    assert np.allclose(out, cm)


def test_load_cell_mass_missing_raises(tmp_path):
    store = str(tmp_path / "empty.zarr")
    xr.Dataset({"dry_mass": ("time", np.ones(5))}, coords={"time": np.arange(5)}).to_zarr(store, mode="w", consolidated=False)
    with pytest.raises(ValueError):
        load_cell_mass(store)
