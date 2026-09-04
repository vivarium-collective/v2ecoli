"""``extract_vectors`` must not materialise the run.

WHY THIS FILE EXISTS, and it is not a style preference.

The extraction used to read its history parquet with a single ``.fetchall()``,
pulling every timestep x every cell x every vector column into Python objects at
once, and then grouping those into a second full copy keyed by cell — roughly 2x
the sweep's vector columns resident.

Measured 2026-08-21 against an 8x16 basal sweep (50 GB of history parquet), at
``generation_lower_bound=0``: 52 GB resident, 63 GB peak, 42.5 GB of 44 GB swap
consumed, the process pinned in uninterruptible I/O wait, and killed rather than
failing on its own. A cache-version bump is exactly the event that triggers it,
because the bump forces a re-extract of every sweep.

⚠ NOT "it could never have completed" -- an earlier draft of this file said that
and the repo's own output tree disproves it. The surviving v1 envelope for this
same sweep records ``extract_seconds: 680.76`` at ``gen_lb=3``, extracted
2026-07-30 at a commit whose ``card_vectors.py`` is the ``fetchall``
implementation, over the same 104 cells. So the old path DID complete this sweep
at gen_lb=3; at gen_lb=0 (+21% rows) it exceeded this machine and was killed.
The honest claim is that peak scales with the RUN, which makes the margin a
property of the hardware rather than of the code -- which is reason enough.

Peak memory must therefore be a function of the ENSEMBLE (cells x features), not
of the RUN (timesteps x cells x features). These tests pin that, and each one
fails against the ``fetchall`` implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library import card_vectors


def _write_sweep(tmp_path, n_cells=3, n_steps=40, width=5, ragged=True):
    """A minimal sweep: one parquet under the layout ``history_files`` globs for.

    ``ragged`` adds one empty-array row per cell so the modal-length rule is
    exercised; the correct implementation drops those rows, so no expected mean
    changes.
    """
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    # Match the real emitter layout: history/experiment_id=…/variant=… — the
    # hive-partition segment `history_files` scopes to.
    d = tmp_path / "exp" / "history" / "experiment_id=e" / "part"
    d.mkdir(parents=True)
    rows = {"lineage_seed": [], "generation": [], "agent_id": []}
    cols = list(card_vectors._VECTOR_COLS)
    for c in cols:
        rows[c] = []
    def _empty_row(cell):
        rows["lineage_seed"].append(0)
        rows["generation"].append(2)
        rows["agent_id"].append(str(cell))
        for c in cols:
            rows[c].append([])

    for cell in range(n_cells):
        if ragged:
            # ⚠ BEFORE the steps as well as after. An empty row only at the END
            # leaves each cell's FIRST row full-width, so a mutant taking the
            # first-seen length instead of the max still passes — measured
            # 2026-08-21. On a real sweep `external_exchange_fluxes`' `[]`
            # default is at least as likely at a cell's first timestep as its
            # last, so bracketing is what makes the fixture representative.
            _empty_row(cell)
        for t in range(n_steps):
            rows["lineage_seed"].append(0)
            rows["generation"].append(2)
            rows["agent_id"].append(str(cell))
            for j, c in enumerate(cols):
                # deterministic, cell-dependent, so a mean is checkable
                rows[c].append([float(cell + j + t) for _ in range(width)])
        if ragged:
            # ⚠ ONE EMPTY-ARRAY ROW PER CELL, AND IT IS LOAD-BEARING.
            #
            # `extract_vectors` keeps only rows matching the column's MODAL
            # (max) length, because `external_exchange_fluxes` emits a `[]`
            # default on some timesteps -- see its docstring. With a uniform
            # fixture min == max, so that rule is satisfied by ANY choice of
            # modal length and the branch is untested: a mutant taking min
            # instead of max passes the whole suite. Measured 2026-08-21.
            #
            # These rows are DROPPED by the correct implementation, so every
            # mean below is unchanged by their presence. Under the min mutant
            # they become the only surviving rows and the node collapses.
            _empty_row(cell)
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


def test_ragged_rows_are_dropped_against_the_MODAL_length_not_the_min(tmp_path):
    """The modal length is the MAX, and taking the min instead must be visible.

    ⚠ Naming, flagged rather than silently propagated: this codebase says
    "modal" throughout but the rule implemented is `if length > col_len[i]`,
    i.e. the MAX. The two differ only if a spuriously LONGER row appears, in
    which case max would drop the entire real panel. Nothing currently pins
    which is intended; this test asserts the implemented rule (max) and both
    ends of the ragged case, so a change to either would surface here.

    `external_exchange_fluxes` emits a `[]` default on some timesteps, so every
    real sweep is ragged and this rule decides whether a node has any data at
    all. Until 2026-08-21 the fixture wrote a uniform width, so min == max and a
    mutant taking the min passed the ENTIRE suite -- the one genuinely subtle
    branch in `extract_vectors`, with no test anywhere (`test_sim_vector_cache`
    stubs the function out).

    Asserted as a POSITIVE and a NEGATIVE, so it cannot pass vacuously: the
    vector must be full-width, and it must equal the mean of the non-empty rows
    only -- which is what the empty rows being dropped MEANS.
    """
    sweep = _write_sweep(tmp_path, n_cells=3, n_steps=4, width=5)
    out = card_vectors.extract_vectors(sweep, generation_lower_bound=0)
    checked = 0

    for j, (_col, (group, name, _u)) in enumerate(card_vectors._VECTOR_COLS.items()):
        node = out[group][name]
        assert len(node["vector"]) == 5, (
            f"{group}.{name} width {len(node['vector'])} != 5 -- the empty rows "
            f"set the modal length, so the min was taken instead of the max")
        assert node["n_cells"] == 3
        # Cell c, step t, column j holds (c + j + t) at every position; the
        # empty rows contribute nothing, so the ensemble mean is over t only.
        expected = sum(c + j + t for c in range(3) for t in range(4)) / 12.0
        checked += 1
        assert node["vector"][0] == pytest.approx(expected), (
            f"{group}.{name} mean {node['vector'][0]} != {expected} -- an empty "
            f"row was averaged in rather than dropped")
    assert checked == len(card_vectors._VECTOR_COLS), (
        f"checked {checked} nodes, expected {len(card_vectors._VECTOR_COLS)} -- "
        f"this test would pass vacuously if extraction returned nothing")


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
