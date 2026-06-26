"""Unit tests for the rnap_state reader in v2ecoli.structural.build."""
import numpy as np
import pytest
from v2ecoli.structural import build


def test_rnap_state_reads_arrays(tmp_path, monkeypatch):
    npz = tmp_path / "v2ecoli_state.npz"
    np.savez(
        npz,
        ids=np.array(["x[c]"]),
        counts=np.array([1]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        rnap_coordinates=np.array([100000, -50000], dtype="i8"),
        rnap_domain_index=np.array([0, 0], dtype="i4"),
        rnap_is_forward=np.array([True, False]),
    )
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("snapshot")
    assert list(st["coordinates"]) == [100000, -50000]
    assert st["is_forward"].dtype == bool and len(st["domain_index"]) == 2


def test_rnap_state_division_file(tmp_path, monkeypatch):
    """division state_source reads v2ecoli_state_division.npz."""
    npz = tmp_path / "v2ecoli_state_division.npz"
    np.savez(
        npz,
        rnap_coordinates=np.array([200000], dtype="i8"),
        rnap_domain_index=np.array([1], dtype="i4"),
        rnap_is_forward=np.array([False]),
    )
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("division")
    assert list(st["coordinates"]) == [200000]
    assert st["domain_index"][0] == 1


def test_rnap_state_empty_fallback(tmp_path, monkeypatch):
    """Missing rnap keys return empty arrays with correct dtypes."""
    npz = tmp_path / "v2ecoli_state.npz"
    np.savez(npz, ids=np.array(["x[c]"]), counts=np.array([1]))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("snapshot")
    assert st["coordinates"].dtype == np.dtype("i8")
    assert st["domain_index"].dtype == np.dtype("i4")
    assert st["is_forward"].dtype == bool
    assert len(st["coordinates"]) == 0


def test_rnap_state_missing_file_fallback(tmp_path, monkeypatch):
    """Missing npz file returns empty arrays without raising."""
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("snapshot")
    assert len(st["coordinates"]) == 0
    assert st["coordinates"].dtype == np.dtype("i8")
