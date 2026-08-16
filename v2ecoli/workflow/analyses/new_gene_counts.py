"""Native port of vEcoli ``ecoli/analysis/multigeneration/new_gene_counts.py``.

New-gene mRNA and protein counts vs time (line plots).  Returns an Altair HTML
view.  Registered as ``"new_gene_counts"`` (scale: ``"multigeneration"``).

Requires a sweep run with the new-gene option enabled.  When no new genes are
present (the typical case), the analysis returns an informational view instead
of charts — matching the source's early return.

v2ecoli adaptations: the listener field orderings normally read via
``field_metadata`` come from sim_data instead —
``listeners__monomer_counts`` ↔ ``translation.monomer_data["id"]`` and
``listeners__rna_counts__mRNA_counts`` ↔ the mRNA TU ids
(``transcription.rna_data["id"][is_mRNA]``).
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    named_idx,
    cast_decimals,
    chart_to_html,
)


def _message_view(msg: str) -> dict:
    return {"view": f'<div class="analysis-view"><h3>New gene counts</h3>'
                    f"<p>{msg}</p></div>"}


class NewGeneCounts(Analysis):
    name = "new_gene_counts"
    scale = "multigeneration"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        cistron = sim_data.process.transcription.cistron_data.struct_array
        monomer = sim_data.process.translation.monomer_data.struct_array
        new_gene_mRNA_ids = cistron[cistron["is_new_gene"]]["id"].tolist()
        mRNA_monomer_id_dict = dict(zip(monomer["cistron_id"], monomer["id"]))
        new_gene_monomer_ids = [
            cast(str, mRNA_monomer_id_dict.get(m)) for m in new_gene_mRNA_ids
        ]

        if len(new_gene_mRNA_ids) == 0 or len(new_gene_monomer_ids) == 0:
            return _message_view(
                "This plot requires simulations with the new-gene option "
                "enabled; no new genes were found in this sweep."
            )

        # Listener field orderings (vEcoli reads these via field_metadata)
        mRNA_ids = sim_data.process.transcription.rna_data["id"][
            sim_data.process.transcription.rna_data["is_mRNA"]
        ].tolist()
        mRNA_idx_dict = {rna[:-3]: i for i, rna in enumerate(mRNA_ids)}
        new_gene_mRNA_indexes = [
            cast(int, mRNA_idx_dict.get(m)) for m in new_gene_mRNA_ids
        ]
        monomer_ids = sim_data.process.translation.monomer_data["id"].tolist()
        monomer_idx_dict = {m: i for i, m in enumerate(monomer_ids)}
        new_gene_monomer_indexes = [
            cast(int, monomer_idx_dict.get(m)) for m in new_gene_monomer_ids
        ]

        new_monomers = named_idx(
            "listeners__monomer_counts", new_gene_monomer_ids,
            [new_gene_monomer_indexes])
        new_mRNAs = named_idx(
            "listeners__rna_counts__mRNA_counts", new_gene_mRNA_ids,
            [new_gene_mRNA_indexes])
        new_gene_data = read_stacked_columns(
            history_sql, [new_monomers, new_mRNAs], conn=conn)
        new_gene_data = cast_decimals(
            pl.DataFrame(new_gene_data).with_columns(
                **{"Time (min)": pl.col("time") / 60})
        )

        mrna_plot = new_gene_data.plot.line(
            x="Time (min)", y=alt.Y(new_gene_mRNA_ids).title("mRNA Counts"),
        ).properties(title="New Gene mRNA Counts")
        protein_plot = new_gene_data.plot.line(
            x="Time (min)", y=alt.Y(new_gene_monomer_ids).title("Protein Counts"),
        ).properties(title="New Gene Protein Counts")
        combined = alt.vconcat(mrna_plot, protein_plot)
        return {"view": chart_to_html(combined, title="New gene counts")}
