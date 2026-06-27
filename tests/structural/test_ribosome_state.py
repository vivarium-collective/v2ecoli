"""Unit tests for build.ribosome_state — Part 1 (no sim required).

Synthesises a minimal npz with the four ribo_* keys and verifies that
ribosome_state returns them under the renamed output keys with correct dtypes.
"""
import numpy as np
import pytest
from v2ecoli.structural import build


def test_ribosome_state_reads_arrays(tmp_path, monkeypatch):
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["x[c]"]),
        counts=np.array([1]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        ribo_mRNA_index=np.array([20, 21], "i8"),
        ribo_pos_on_mRNA=np.array([0, 300], "i8"),
        ribo_peptide_length=np.array([0, 100], "i8"),
        ribo_protein_index=np.array([5, 6], "i8"),
    )
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.ribosome_state("snapshot")
    assert list(st["mRNA_index"]) == [20, 21]
    assert list(st["pos_on_mRNA"]) == [0, 300] and st["peptide_length"].dtype == np.int64
