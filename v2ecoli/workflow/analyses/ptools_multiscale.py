"""Multi-scale registrations of the PathwayTools (ptools) TSV analyses.

In vEcoli the ptools analyses live at three scales (``single``,
``multigeneration``, ``multiseed``) as byte-identical source files — the
difference is only which cells the ``history_sql`` covers and how ``read_outputs``
aggregates them by ``time``.

**Multigeneration** (:class:`_MultigenMixin`): a single-daughter lineage has
exactly one cell per generation.  vEcoli's absolute, monotonic ``time`` makes
``read_outputs``' ``GROUP BY time`` an identity (no row merges); v2ecoli's
``global_time`` resets each generation, so we rebuild the absolute axis with
:func:`_helpers.cumulative_time_history` and then reuse the single-scale
``analyze`` verbatim.  Result: the TSV spans the whole lineage with correct
minute checkpoints (e.g. 0m … 88m across two generations).

**Multiseed** (:class:`_MultiseedMixin`): multiple seeds share the same
absolute ``time`` value.  v2ecoli's ``groupby("time").sum()`` over Python-list
columns (``bulk__id``, ``bulk__count``, flux arrays) *concatenates* instead of
adding element-wise — wrong.  This mixin overrides ``_do_read_outputs()`` to
fetch the raw per-seed rows and apply :func:`_helpers.collapse_cross_seed`:
string-list id columns are first-preserved (with length assertion), numeric
list/array columns are element-wise numpy-summed (with shape assertion), and
scalar columns are plain-summed.  The single-scale ``analyze`` body then runs
unchanged on the aggregated DataFrame.
"""

from __future__ import annotations

from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    cumulative_time_history,
    collapse_cross_seed,
    ptools_time_series_query,
)
from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna
from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins


# ---------------------------------------------------------------------------
# _MultigenMixin — absolute time axis for single-daughter lineages
# ---------------------------------------------------------------------------

class _MultigenMixin:
    """Rewrite history to an absolute time axis, then run the single analyze."""

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        abs_sql = cumulative_time_history(history_sql)
        return super().analyze(
            conn=conn,
            history_sql=abs_sql,
            sim_data=sim_data,
            variant_metadata=variant_metadata,
            **ctx,
        )


class PtoolsRnaMultigeneration(_MultigenMixin, PtoolsRna):
    name = "ptools_rna_multigeneration"
    scale = "multigeneration"


class PtoolsRxnsMultigeneration(_MultigenMixin, PtoolsRxns):
    name = "ptools_rxns_multigeneration"
    scale = "multigeneration"


class PtoolsProteinsMultigeneration(_MultigenMixin, PtoolsProteins):
    name = "ptools_proteins_multigeneration"
    scale = "multigeneration"


# ---------------------------------------------------------------------------
# _MultiseedMixin — cross-seed element-wise aggregation
# ---------------------------------------------------------------------------

class _MultiseedMixin:
    """Override _do_read_outputs to collapse multiple seeds element-wise.

    Multiple seeds share the same absolute ``time`` value, so the per-seed
    rows must be aggregated before the single-scale ``analyze`` body runs.
    :func:`collapse_cross_seed` handles this correctly:

    * ``bulk__id`` (and any other ``id_cols``) — first-preserved (same across
      seeds; length-mismatch raises ``ValueError``).
    * list/array columns (``bulk__count``, FBA flux arrays, per-transcript
      lists) — element-wise numpy sum (shape-mismatch raises ``ValueError``).
    * scalar columns (``active_ribosome``, ``oriC``, ``active_RNAP``) — plain
      scalar sum.

    The cross-seed aggregated DataFrame is then passed verbatim to the
    inherited single-scale ``analyze`` body.
    """

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
        filter_clause="",
    ):
        """Fetch raw per-seed rows and collapse to one row per time."""
        if columns is None:
            # Fallback: should not happen in practice (analyze always passes
            # explicit columns), but guard against bare calls.
            raise ValueError("_MultiseedMixin._do_read_outputs requires explicit columns")

        # Build the same SQL used by the single-scale read_outputs — but do NOT
        # apply groupby.sum() here; collapse_cross_seed handles aggregation.
        # ``filter_clause`` only restricts which rows reach that aggregation
        # (see ptools_time_series_query's docstring); it adds no aggregation
        # of its own, so collapse_cross_seed's own cross-seed math is
        # untouched.
        query_sql = ptools_time_series_query(columns, history_sql, filter_clause)
        raw_df = conn.sql(query_sql).df()
        return collapse_cross_seed(raw_df, id_cols=frozenset({"bulk__id"}))


class PtoolsRnaMultiseed(_MultiseedMixin, PtoolsRna):
    name = "ptools_rna_multiseed"
    scale = "multiseed"


class PtoolsRxnsMultiseed(_MultiseedMixin, PtoolsRxns):
    name = "ptools_rxns_multiseed"
    scale = "multiseed"


class PtoolsProteinsMultiseed(_MultiseedMixin, PtoolsProteins):
    name = "ptools_proteins_multiseed"
    scale = "multiseed"
