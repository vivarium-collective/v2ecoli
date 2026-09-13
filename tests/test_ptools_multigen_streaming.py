"""Streaming (per-generation) reads for ``_MultigenMixin``.

The multigeneration ptools (``ptools_{rna,rxns,proteins,metabolites}_multigeneration``)
run the SINGLE-scale analyze body over a whole lineage's cumulative-time history.
That body reads every generation × timestep row — including the wide ``DOUBLE[]``
list columns — in one ``conn.sql(...).df()`` and ``np.stack``s it, then
``consolidate_timepoints`` reduces to ONE column per generation. On a 20-generation
sweep that single frame is what fills the analysis container's temp-spill DISK
(the wide per-category reads, #786-class sizing) — DISK, not RAM, so no memory
class fixes it.

#789 streamed the *multiseed* collapse (``_MultiseedCollapseMixin``, one class:
PtoolsOverviewMultiseed) but does NOT touch ``_MultigenMixin``. This is its
per-generation analog: because ``per_generation=True`` already reduces each
generation to a single output window, and that reduction is per-generation
independent (``consolidate_timepoints(generations=)`` = mean/sum over each
generation's own ticks), the read can stream one generation at a time and keep
peak resident at a single generation's rows.

Two properties pin the fix:

* **Equivalence** — the multigeneration table is unchanged (streamed == whole-frame),
  asserted here on a controlled history whose per-generation means are known.
* **Memory bound** — no single ``.df()`` materialises more than one generation's
  rows. This FAILS on the current one-shot reader (one ``.df()`` over all
  generations) and PASSES once the read streams per generation.
"""

from __future__ import annotations

from types import SimpleNamespace

import duckdb

from bigraph_schema import allocate_core
from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
from v2ecoli.workflow.analyses.ptools_multiscale import _MultigenMixin


FLUX_COL = "listeners__fba_results__base_reaction_fluxes"
RXN_IDS = ["RXN_A", "RXN_B", "RXN_C"]        # flux width = 3
N_GEN = 3                                     # gens 0,1,2 (skip_n_gens=1 drops gen 0)
TICKS_PER_GEN = 4


class _PtoolsRxnsMultigen(_MultigenMixin, PtoolsRxns):
    """The registered PtoolsRxnsMultigeneration, redeclared locally so the test does
    not depend on registration order."""
    name = "ptools_rxns_multigeneration_test"
    scale = "multigeneration"


def _fake_sim_data():
    return SimpleNamespace(
        process=SimpleNamespace(
            metabolism=SimpleNamespace(base_reaction_ids=list(RXN_IDS))
        )
    )


def _make_history():
    """Single-lineage, N_GEN generations × TICKS_PER_GEN ticks. Each generation's
    flux is CONSTANT across its ticks (gen g -> [g+1, 2*(g+1), 3*(g+1)]) so the
    per-generation mean is exactly that vector — a clean equivalence target.
    global_time is absolute (parquet convention) and monotonic across generations.
    """
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE hist ("
        "  experiment_id VARCHAR, variant BIGINT, lineage_seed BIGINT,"
        "  generation BIGINT, agent_id VARCHAR, global_time DOUBLE,"
        f'  "{FLUX_COL}" DOUBLE[]'
        ")"
    )
    t = 0.0
    for g in range(N_GEN):
        base = g + 1
        flux = [float(base), float(2 * base), float(3 * base)]
        agent = "0" * (g + 1)               # single-daughter lineage chain
        for _ in range(TICKS_PER_GEN):
            conn.execute(
                "INSERT INTO hist VALUES ('exp', 0, 0, ?, ?, ?, ?)",
                [g, agent, t, flux],
            )
            t += 1.0
    return conn, "SELECT * FROM hist"


class _CountingRelation:
    def __init__(self, rel, sink):
        self._rel, self._sink = rel, sink

    def df(self):
        frame = self._rel.df()
        self._sink.append(len(frame))
        return frame

    def __getattr__(self, name):
        return getattr(self._rel, name)


class _CountingConn:
    """Records the row count of every ``.df()`` so a test can assert no single
    frame holds more than one generation's rows."""

    def __init__(self, conn):
        self._conn = conn
        self.df_row_counts: list[int] = []

    def sql(self, query):
        return _CountingRelation(self._conn.sql(query), self.df_row_counts)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _run(conn):
    step = _PtoolsRxnsMultigen({}, core=allocate_core())
    return step.analyze(conn=conn, history_sql="SELECT * FROM hist",
                        sim_data=_fake_sim_data())


def test_multigen_table_matches_expected_per_generation_means():
    """Correctness guard across the refactor: with skip_n_gens=1 the first
    generation is dropped, leaving gens 1 and 2 whose per-generation flux means
    are [2,4,6] and [3,6,9]. The rendered table is |flux| with reactions as rows."""
    conn, _ = _make_history()
    out = _run(conn)
    tsv = out["data"]["tsv"]
    # Parse the reaction rows out of the TSV (index col '$', then two gen columns).
    rows = {}
    for line in tsv.strip().splitlines()[1:]:
        parts = line.split("\t")
        rows[parts[0]] = [float(x) for x in parts[1:]]
    assert set(rows) == set(RXN_IDS), f"unexpected reaction rows: {sorted(rows)}"
    # gen1 mean = [2,4,6], gen2 mean = [3,6,9]; each reaction row is (gen1, gen2).
    assert rows["RXN_A"] == [2.0, 3.0], rows["RXN_A"]
    assert rows["RXN_B"] == [4.0, 6.0], rows["RXN_B"]
    assert rows["RXN_C"] == [6.0, 9.0], rows["RXN_C"]


def test_multigen_never_materializes_more_than_one_generation():
    """The memory-bound property. FAILS on the one-shot reader (one .df() over all
    N_GEN*TICKS_PER_GEN rows) and PASSES once the read streams per generation."""
    conn, _ = _make_history()
    spy = _CountingConn(conn)
    _run(spy)
    assert spy.df_row_counts, "expected at least one .df() materialisation"
    assert max(spy.df_row_counts) <= TICKS_PER_GEN, (
        f"a single .df() materialised {max(spy.df_row_counts)} rows, more than one "
        f"generation's {TICKS_PER_GEN}: the multigen read is not streaming per "
        f"generation (this is the disk-spill the fix removes)."
    )
