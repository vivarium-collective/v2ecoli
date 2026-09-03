"""``extract_vectors`` must not assume every observable column is present.

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

2. **That every file in the glob has the same schema.** DuckDB binds
   ``read_parquet`` against the FIRST file unless told otherwise, so a pooled
   directory silently loses a group (or raises) depending on which file sorts
   first. ``union_by_name=true`` is what makes the probe describe the sweep
   rather than one file of it.

⛔ **A cell-completeness rule was attempted in this PR and WITHDRAWN.** A
row-count heuristic (drop cells far below the longest) was found, by independent
review, to do nothing on a real in-tree sweep whose emit cadence is coarse — the
partial cell was 14% of the longest and sailed through, while the node
affirmatively reported that nothing had been dropped. Whether a stub falls under
a fixed fraction is a property of the producing run, which is the very thing this
module is trying to stop assuming. The correct signal is a per-cell ``divided``
flag; it is not currently an emitted parquet column. Tracked as follow-up.
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
    # ⚠ REAL hive layout, not a flat "part" directory. `extract_vectors` reads
    # with `hive_partitioning=true`, and `history_files` globs
    # `<sweep>/**/history/**/*.pq` — a fixture that skips the `key=value`
    # directories exercises neither. Matching the shape a runner actually emits
    # is what makes a green fixture evidence about real sweeps.
    d = (tmp_path / "exp" / "history" / "experiment_id=exp" / "variant=0"
         / "lineage_seed=0" / "generation=2" / "agent_id=00")
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
    sweep = str(tmp_path / "exp")
    # ⛔ THE FIXTURE ASSERTS ITSELF. Without this, a fixture whose files the
    # reader cannot find produces `{}` — which every test below then reports as
    # a wrong RESULT rather than an absent INPUT. Three tests failed that way in
    # CI while passing locally, and the messages (`AssertionError: {}`,
    # `KeyError: 'omics'`) pointed at the code under test rather than at the
    # fixture. A fixture that cannot prove it built something is not a fixture.
    from v2ecoli.library.sweep_io import history_files
    found = history_files(sweep)
    assert found, f"fixture wrote no discoverable history parquet under {sweep}"
    return sweep


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




def test_a_column_missing_from_only_the_FIRST_file_is_still_found(tmp_path):
    """⛔ DuckDB binds the glob against the first file unless union_by_name is set.

    The two failure modes are asymmetric and only one is loud: a first file
    LACKING a column the others have omits the group SILENTLY; a first file
    HAVING one the others lack raises. This pins the silent one, which is the
    dangerous direction — a whole observable group vanishing with no error."""
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    d = (tmp_path / "exp" / "history" / "experiment_id=exp" / "variant=0"
         / "lineage_seed=0" / "generation=2" / "agent_id=00")
    d.mkdir(parents=True)
    n = 20
    base = {"lineage_seed": [0]*n, "generation": [2]*n, "agent_id": ["00"]*n}
    narrow = dict(base)
    for c in ("listeners__rna_counts__mRNA_cistron_counts", "listeners__monomer_counts"):
        narrow[c] = [[1.0, 1.0]]*n
    wide = dict(narrow)
    wide["listeners__fba_results__external_exchange_fluxes"] = [[2.0, 2.0]]*n
    # "0.pq" sorts first and is the NARROW one
    pyarrow.parquet.write_table(pq.table(narrow), d / "0.pq")
    pyarrow.parquet.write_table(pq.table(wide), d / "1.pq")

    from v2ecoli.library.sweep_io import history_files
    assert len(history_files(str(tmp_path / "exp"))) == 2, "fixture wrote <2 files"
    out = card_vectors.extract_vectors(str(tmp_path / "exp"))
    assert "fluxes" in out, (
        "the exchange group vanished because only the first file was inspected")
