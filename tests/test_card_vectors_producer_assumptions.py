"""``extract_vectors`` must not assume things about the run that PRODUCED a sweep.

WHY THIS FILE EXISTS.

Two properties of a sweep were, until now, assumed rather than checked, and both
assumptions held for every sweep the function had been used on — so neither was
wrong in a way any existing test could see:

1. **That every observable column is present.** Different metabolism processes
   write different exchange leaves: ``metabolism.py`` writes
   ``listeners.fba_results.external_exchange_fluxes``; ``metabolism_redux`` does
   not (it writes ``estimated_exchange_dmdt``). The same divergence is already
   documented and guarded, for the ``gdcw`` basis, in
   ``library/vivarium_ecoli_engine.py``. Here it surfaced as a DuckDB binder
   error that named a column rather than the cause, and took the omics groups
   down with it even though those columns were present.

2. **That one ``(lineage_seed, generation, agent_id)`` group is one cell cycle.**
   A runner that hand-drives its generations emits BOTH daughters at division
   and then follows only one; the abandoned daughter leaves a short birth stub
   in its own ``agent_id`` partition. Sweeps produced with the workflow's
   ``single_daughters`` setting contain no stubs, which is why this held.

★ **The generalisable point, and the reason these are one file rather than two:**
both are assumptions about the PRODUCER that were satisfied by CONFIGURATION and
asserted NOWHERE. The consequence of (2) in particular is silent — a stub is a
few dozen ticks of newborn state averaged beside a full cell cycle, so it moves
every ensemble statistic toward the newborn without failing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library import card_vectors


def _write(tmp_path, cells, cols=None):
    """Write one sweep. ``cells`` is ``[(agent_id, n_steps), ...]``.

    Values are constant within a cell and differ between cells, so an ensemble
    mean states plainly WHICH cells went into it.
    """
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    cols = list(card_vectors._VECTOR_COLS) if cols is None else list(cols)
    d = tmp_path / "exp" / "history" / "part"
    d.mkdir(parents=True)
    rows = {"lineage_seed": [], "generation": [], "agent_id": []}
    for c in cols:
        rows[c] = []
    for value, (agent, n_steps) in enumerate(cells, start=1):
        for _ in range(n_steps):
            rows["lineage_seed"].append(0)
            rows["generation"].append(2)
            rows["agent_id"].append(agent)
            for c in cols:
                rows[c].append([float(value)] * 3)
    pyarrow.parquet.write_table(pq.table(rows), d / "0.pq")
    return str(tmp_path / "exp")


# --------------------------------------------------------------------------
# 1. A column this sweep does not have
# --------------------------------------------------------------------------

def test_an_absent_column_omits_its_group_and_leaves_the_others_extractable(tmp_path):
    """The case a redux-metabolism sweep actually presents."""
    present = [
        "listeners__rna_counts__mRNA_cistron_counts",
        "listeners__monomer_counts",
    ]
    out = card_vectors.extract_vectors(_write(tmp_path, [("0", 10)], cols=present))
    assert sorted(out) == ["omics"], out
    assert sorted(out["omics"]) == ["proteome", "transcriptome"]


def test_an_absent_column_is_NOT_reported_as_zeros(tmp_path):
    """⛔ The whole point. A zero-filled exchange vector is indistinguishable
    from a cell exchanging nothing — an error silently becoming a result."""
    present = [
        "listeners__rna_counts__mRNA_cistron_counts",
        "listeners__monomer_counts",
    ]
    out = card_vectors.extract_vectors(_write(tmp_path, [("0", 10)], cols=present))
    assert "fluxes" not in out, (
        "the absent exchange group must be OMITTED, not emitted as zeros")


def test_a_sweep_with_no_observable_columns_raises(tmp_path):
    """Degrading to ``{}`` here would hand a caller an empty ensemble that looks
    like a run which simply had nothing to say."""
    with pytest.raises(ValueError, match="no observable columns"):
        card_vectors.extract_vectors(
            _write(tmp_path, [("0", 10)], cols=["listeners__mass__dry_mass"]))


# --------------------------------------------------------------------------
# 2. Rows that are not a cell cycle
# --------------------------------------------------------------------------

def test_birth_stubs_are_excluded_from_the_ensemble(tmp_path):
    """One full cell (value 1) and one 2-tick stub (value 2).

    The mean is the property under test, not ``n_cells``: including the stub
    gives 1.5, excluding it gives 1.0. A mutant that counts correctly but still
    averages the stub in passes an ``n_cells`` assertion and fails this one."""
    out = card_vectors.extract_vectors(_write(tmp_path, [("00", 400), ("01", 2)]))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 1
    assert node["n_cells_excluded_partial"] == 1
    assert np.allclose(node["vector"], 1.0), (
        "the stub was averaged into the ensemble mean")


def test_the_excluded_count_is_reported_rather_than_silent(tmp_path):
    """A provenance panel printing ``n_cells`` alone cannot distinguish an
    ensemble that never had more cells from one whose extras were dropped."""
    out = card_vectors.extract_vectors(_write(tmp_path, [("00", 400), ("01", 2)]))
    assert out["omics"]["transcriptome"]["n_cells_excluded_partial"] == 1


def test_a_sweep_of_only_full_cells_excludes_NOTHING(tmp_path):
    """The paired control. Without it, ``exclude everything but the longest
    cell`` would pass every other test in this file."""
    out = card_vectors.extract_vectors(
        _write(tmp_path, [("00", 400), ("000", 380), ("0000", 420)]))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 3
    assert node["n_cells_excluded_partial"] == 0
    assert np.allclose(node["vector"], 2.0)


def test_stubs_are_excluded_EVEN_WHEN_THEY_ARE_THE_MAJORITY(tmp_path):
    """⛔⛔ THE TEST THAT CAUGHT THE FIRST IMPLEMENTATION OF THIS RULE.

    The rule was first written against the sweep's MEDIAN rows-per-cell. On a
    real 4-generation sweep the per-cell row counts were
    ``25, 25, 29, 31, 45, 3217, 3333, 3511`` — five stubs to three cells — so
    the median was 38, the floor 3.8, and **all eight were admitted**. The rule
    failed silently in exactly the case it exists for, and no synthetic fixture
    with a stub minority would have shown it.

    ★ A robust statistic is the wrong tool when the contaminant can be the
    majority. Scaling off the longest cell is what makes this hold."""
    cells = [("a", 25), ("b", 25), ("c", 29), ("d", 31), ("e", 45),
             ("f", 3217), ("g", 3333), ("h", 3511)]
    out = card_vectors.extract_vectors(_write(tmp_path, cells))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 3, (
        "stubs in the majority dragged the scale down and were admitted")
    assert node["n_cells_excluded_partial"] == 5


def test_membership_is_the_same_cell_set_for_every_group(tmp_path):
    """Deciding membership per column would let two halves of one comparison
    describe different cells, with nothing in the output saying so."""
    out = card_vectors.extract_vectors(_write(tmp_path, [("00", 400), ("01", 2)]))
    counts = {(g, n): node["n_cells"]
              for g, d in out.items() for n, node in d.items()}
    assert len(set(counts.values())) == 1, counts
