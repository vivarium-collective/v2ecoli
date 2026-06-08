"""Multigeneration registrations of the PathwayTools (ptools) TSV analyses.

In vEcoli the ptools analyses live at three scales (``single``,
``multigeneration``, ``multiseed``) as byte-identical source files — the
difference is only which cells the ``history_sql`` covers and how ``read_outputs``
aggregates them by ``time``.

**Multigeneration** (here): a single-daughter lineage has exactly one cell per
generation.  vEcoli's absolute, monotonic ``time`` makes ``read_outputs``'
``GROUP BY time`` an identity (no row merges); v2ecoli's ``global_time`` resets
each generation, so we rebuild the absolute axis with
:func:`_helpers.cumulative_time_history` and then reuse the single-scale
``analyze`` verbatim.  Result: the TSV spans the whole lineage with correct
minute checkpoints (e.g. 0m … 88m across two generations).

**Multiseed** is intentionally NOT ported here — see the porting progress note.
At multiseed scale multiple seeds share each ``(generation, time)``, so
``read_outputs`` is meant to sum the list columns element-wise across seeds.
v2ecoli's pandas ``groupby.sum`` over the per-row ``bulk__id`` (string lists)
and ``bulk__count`` (python lists) columns *concatenates* instead of adding,
so the single read path cannot be reused; a dedicated cross-seed read
(``first(bulk__id)`` + element-wise ndarray sum) is needed.  Recorded BLOCKED.
"""

from __future__ import annotations

from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import cumulative_time_history
from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna
from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins


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
