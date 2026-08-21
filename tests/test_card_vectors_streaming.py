"""``extract_vectors`` must not materialise the run.

WHY THIS FILE EXISTS, and it is not a style preference.

The extraction used to read its history parquet with a single ``.fetchall()``,
pulling every timestep x every cell x every vector column into Python objects at
once, and then grouping those into a second full copy keyed by cell — roughly 2x
the sweep's vector columns resident.

Measured 2026-08-21 against an 8x16 basal sweep (50 GB of history parquet): 52 GB
resident, 63 GB peak, 42.5 GB of 44 GB swap consumed, the process pinned in
uninterruptible I/O wait. It could not have completed, and it took the machine
down rather than failing on its own. A cache-version bump is exactly the event
that triggers it, because the bump forces a re-extract of every sweep.

Peak memory must therefore be a function of the ENSEMBLE (cells x features), not
of the RUN (timesteps x cells x features). These tests pin that, and each one
fails against the ``fetchall`` implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library import card_vectors


def _write_sweep(tmp_path, n_cells=3, n_steps=40, width=5):
    """A minimal sweep: one parquet under the layout ``history_files`` globs for."""
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    d = tmp_path / "exp" / "history" / "part"
    d.mkdir(parents=True)
    rows = {"lineage_seed": [], "generation": [], "agent_id": []}
    cols = list(card_vectors._VECTOR_COLS)
    for c in cols:
        rows[c] = []
    for cell in range(n_cells):
        for t in range(n_steps):
            rows["lineage_seed"].append(0)
            rows["generation"].append(2)
            rows["agent_id"].append(str(cell))
            for j, c in enumerate(cols):
                # deterministic, cell-dependent, so a mean is checkable
                rows[c].append([float(cell + j + t) for _ in range(width)])
    pyarrow.parquet.write_table(pq.table(rows), d / "0.pq")
    return str(tmp_path / "exp")


class _SpyRelation:
    """Wraps a DuckDB relation and records how the result was consumed."""

    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    def fetchmany(self, size):
        self._log["fetchmany"] += 1
        self._log["sizes"].add(size)
        return self._inner.fetchmany(size)

    def fetchall(self):
        self._log["fetchall"] += 1
        raise AssertionError(
            "extract_vectors called fetchall() — that materialises the whole run "
            "and is the failure this module documents")

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def spy(monkeypatch):
    log = {"fetchmany": 0, "fetchall": 0, "sizes": set()}
    # Bind the REAL function before patching: the patch replaces this very name,
    # so re-importing it inside the wrapper would call the wrapper.
    from v2ecoli.library.sweep_io import connect_for as _real_connect_for

    def patched(sweep_dir):
        con = _real_connect_for(sweep_dir)

        class _Con:
            def sql(self, q):
                return _SpyRelation(con.sql(q), log)

            def __getattr__(self, n):
                return getattr(con, n)

        return _Con()

    monkeypatch.setattr("v2ecoli.library.sweep_io.connect_for", patched)
    return log


def test_reads_in_batches_rather_than_all_at_once(tmp_path, monkeypatch, spy):
    """The discriminating test: fails against ``fetchall``, loudly.

    With the batch size forced below the row count, a streaming implementation
    must call ``fetchmany`` more than once. A materialising one calls
    ``fetchall`` and the spy raises.
    """
    monkeypatch.setattr(card_vectors, "_FETCH_BATCH_ROWS", 7)
    sweep = _write_sweep(tmp_path, n_cells=3, n_steps=40)

    out = card_vectors.extract_vectors(sweep, 0)

    assert spy["fetchall"] == 0
    assert spy["fetchmany"] > 1, (
        f"120 rows at a batch size of 7 should take many fetches, got "
        f"{spy['fetchmany']} — the read is not streaming")
    assert out, "extraction produced nothing, so the assertions above are vacuous"


def test_batching_does_not_change_the_numbers(tmp_path, monkeypatch):
    """A batch boundary must not split a cell's time-mean.

    The running sum is keyed by cell, so a cell spanning batches has to
    accumulate across them. Comparing a tiny batch against a batch larger than
    the whole input is what catches an accumulator that resets per batch —
    which would be invisible in any single-batch test.
    """
    sweep = _write_sweep(tmp_path, n_cells=3, n_steps=40)

    monkeypatch.setattr(card_vectors, "_FETCH_BATCH_ROWS", 3)
    many = card_vectors.extract_vectors(sweep, 0)
    monkeypatch.setattr(card_vectors, "_FETCH_BATCH_ROWS", 100_000)
    one = card_vectors.extract_vectors(sweep, 0)

    assert set(many) == set(one) and many
    for g in one:
        for n in one[g]:
            assert many[g][n]["n_cells"] == one[g][n]["n_cells"]
            np.testing.assert_allclose(
                many[g][n]["vector"], one[g][n]["vector"], rtol=1e-12,
                err_msg=f"{g}.{n} changed with batch size — the accumulator is "
                        f"not carrying across batches")
            np.testing.assert_allclose(
                many[g][n]["per_cell"], one[g][n]["per_cell"], rtol=1e-12)


def test_cell_order_is_first_appearance_not_merely_self_consistent(tmp_path, monkeypatch):
    """``per_cell`` row order is part of the contract; pin it ABSOLUTELY.

    ⚠ An earlier version of this test compared two batch sizes against each
    other and passed against a mutant that reversed the order — both runs were
    wrong the same way. Comparing an implementation to itself cannot detect a
    contract it never satisfied. So this asserts the values.

    The fixture is built so cell ``c``'s column ``j`` is ``c + j + t`` at step
    ``t``; its time-mean is therefore ``c + j + (n_steps-1)/2``, strictly
    increasing in ``c``. Row ``i`` of ``per_cell`` must be cell ``i``.

    Nothing else in the pipeline records which row is which cell, so a
    reordering here silently re-labels every per-cell sample downstream.
    """
    n_cells, n_steps, width = 4, 10, 5
    sweep = _write_sweep(tmp_path, n_cells=n_cells, n_steps=n_steps, width=width)
    monkeypatch.setattr(card_vectors, "_FETCH_BATCH_ROWS", 3)

    out = card_vectors.extract_vectors(sweep, 0)
    t_mean = (n_steps - 1) / 2

    checked = 0
    for j, (col, (group, name, _u)) in enumerate(card_vectors._VECTOR_COLS.items()):
        node = out.get(group, {}).get(name)
        if node is None:
            continue
        per_cell = np.asarray(node["per_cell"], float)
        assert per_cell.shape == (n_cells, width), f"{group}.{name} shape"
        expected = np.array([[c + j + t_mean] * width for c in range(n_cells)])
        np.testing.assert_allclose(
            per_cell, expected, rtol=1e-12,
            err_msg=f"{group}.{name}: per_cell rows are not in first-appearance "
                    f"cell order (row i must be cell i)")
        checked += 1
    assert checked, "no column was checked, so this test asserted nothing"
