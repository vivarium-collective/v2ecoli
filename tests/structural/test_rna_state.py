"""Unit tests for rna_state reader + rnap_state unique_index in v2ecoli.structural.build."""
import numpy as np
import pytest
from v2ecoli.structural import build


def test_rna_state_reads_arrays(tmp_path, monkeypatch):
    np.savez(tmp_path / "v2ecoli_state.npz",
        ids=np.array(["x[c]"]), counts=np.array([1]), volume=np.array(1.0),
        n_chromosomes=np.array(1), fork_fraction=np.array(0.45), division_progress=np.array(0.0),
        rnap_coordinates=np.array([100000], "i8"), rnap_domain_index=np.array([0], "i4"),
        rnap_is_forward=np.array([True]), rnap_unique_index=np.array([7], "i8"),
        rna_unique_index=np.array([20, 21], "i8"), rna_RNAP_index=np.array([7, -1], "i8"),
        rna_transcript_length=np.array([850, 1200], "i8"), rna_is_mRNA=np.array([True, True]),
        rna_is_full_transcript=np.array([False, True]), rna_TU_index=np.array([3, 4], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    assert list(build.rnap_state("snapshot")["unique_index"]) == [7]
    st = build.rna_state("snapshot")
    assert list(st["RNAP_index"]) == [7, -1]
    assert st["is_mRNA"].dtype == bool and list(st["transcript_length"]) == [850, 1200]


def test_rna_state_division_file(tmp_path, monkeypatch):
    """division state_source reads v2ecoli_state_division.npz."""
    np.savez(tmp_path / "v2ecoli_state_division.npz",
        rnap_coordinates=np.array([200000], "i8"), rnap_domain_index=np.array([1], "i4"),
        rnap_is_forward=np.array([False]), rnap_unique_index=np.array([99], "i8"),
        rna_unique_index=np.array([50], "i8"), rna_RNAP_index=np.array([99], "i8"),
        rna_transcript_length=np.array([300], "i8"), rna_is_mRNA=np.array([False]),
        rna_is_full_transcript=np.array([False]), rna_TU_index=np.array([1], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("division")
    assert list(st["unique_index"]) == [99]
    rna = build.rna_state("division")
    assert list(rna["RNAP_index"]) == [99]


def test_rna_state_empty_fallback(tmp_path, monkeypatch):
    """Missing rna keys return empty arrays with correct dtypes."""
    np.savez(tmp_path / "v2ecoli_state.npz", ids=np.array(["x[c]"]), counts=np.array([1]))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rna_state("snapshot")
    assert st["unique_index"].dtype == np.dtype("i8")
    assert st["RNAP_index"].dtype == np.dtype("i8")
    assert st["transcript_length"].dtype == np.dtype("i8")
    assert st["is_mRNA"].dtype == bool
    assert st["is_full_transcript"].dtype == bool
    assert st["TU_index"].dtype == np.dtype("i8")
    assert len(st["unique_index"]) == 0


def test_rna_state_missing_file_fallback(tmp_path, monkeypatch):
    """Missing npz returns empty arrays without raising."""
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rna_state("snapshot")
    assert len(st["unique_index"]) == 0
    assert st["unique_index"].dtype == np.dtype("i8")


def test_rnap_unique_index_empty_fallback(tmp_path, monkeypatch):
    """Missing rnap_unique_index key returns empty i8 array."""
    np.savez(tmp_path / "v2ecoli_state.npz",
        rnap_coordinates=np.array([100000], "i8"),
        rnap_domain_index=np.array([0], "i4"),
        rnap_is_forward=np.array([True]))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("snapshot")
    assert st["unique_index"].dtype == np.dtype("i8")
    assert len(st["unique_index"]) == 0
