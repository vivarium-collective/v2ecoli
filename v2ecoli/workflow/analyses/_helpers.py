"""Shared helpers for native vEcoli analysis ports (the altair/polars family).

vEcoli's plot analyses lean on ``ecoli.library.parquet_emitter`` helpers
(``read_stacked_columns``, ``num_cells``, ``skip_n_gens``, ``named_idx``,
``ndidx_to_duckdb_expr``) and produce Altair charts saved to HTML.  Two things
differ in v2ecoli's parquet schema and must be bridged natively here:

  * **``time`` column** — vEcoli emits a ``time`` column; v2ecoli emits
    ``global_time``.  ``read_stacked_columns`` (and column expressions that
    reference ``time``) need it, so :func:`aliased_history` wraps any
    ``history_sql`` to expose ``global_time`` *also* as ``time``.

  * **``bulk`` column** — vEcoli emits a single ``bulk`` list column ordered to
    match ``sim_data`` bulk order.  v2ecoli emits ``bulk__id`` + ``bulk__count``
    (per-row lists in a parquet ordering that may differ from sim_data).
    :func:`bulk_field_ids` returns the parquet bulk ordering (the equivalent of
    ``field_metadata(conn, config_sql, "bulk")``), and
    :func:`bulk_count_idx_expr` builds a ``named_idx`` expression over
    ``bulk__count`` for a set of molecule ids — failing loudly if any id is
    absent (never silently dropping).

The pure SQL-expression builders ``named_idx`` and ``ndidx_to_duckdb_expr`` are
schema-agnostic, so they are re-exported from the vEcoli module unchanged.

Altair views are serialized with :func:`chart_to_html` (``chart.to_html()``),
the view-output analogue of ``chart.save(path)``.
"""

from __future__ import annotations

from typing import Optional

import duckdb

# Pure SQL-expression builders — no schema assumptions, reuse as-is.
from ecoli.library.parquet_emitter import named_idx, ndidx_to_duckdb_expr  # noqa: F401

# Shim B (see _shims): active-ribosome scalar, aliased to the vEcoli column name
# so ported code can keep referencing the original unique_molecule_counts name.
ACTIVE_RIBOSOME_AS_UMC = (
    "list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)"
    " AS listeners__unique_molecule_counts__active_ribosome"
)


def aliased_history(history_sql: str) -> str:
    """Wrap ``history_sql`` so ``global_time`` is also exposed as ``time``."""
    return f"SELECT *, global_time AS time FROM ({history_sql})"


_ID_COLS = "experiment_id, variant, lineage_seed, generation, agent_id, time"


def read_stacked_columns(
    history_sql: str,
    columns: list[str],
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    order_results: bool = True,
    success_sql: Optional[str] = None,
    remove_first: bool = False,
):
    """Native port of ``ecoli.library.parquet_emitter.read_stacked_columns``.

    Selects ``columns`` (raw names or DuckDB expressions) plus the cell-id
    columns and ``time`` from ``history_sql``.  ``time`` is synthesised from
    ``global_time`` via :func:`aliased_history`, so column expressions may
    reference ``time`` freely.  Returns a polars DataFrame when ``conn`` is
    given, otherwise the SQL string (for use as a subquery).
    """
    base = aliased_history(history_sql)
    cols = ", ".join(columns)
    sql = f"SELECT {cols}, {_ID_COLS} FROM ({base})"
    if success_sql:
        sql = (
            f"SELECT * FROM ({sql}) SEMI JOIN ({success_sql}) "
            "USING (experiment_id, variant, lineage_seed, generation, agent_id)"
        )
    if remove_first:
        sql = f"""
            SELECT * FROM ({sql})
            ANTI JOIN (
                SELECT experiment_id, variant, lineage_seed, generation,
                    agent_id, MIN(global_time) AS time
                FROM ({history_sql})
                GROUP BY experiment_id, variant, lineage_seed, generation,
                    agent_id
            ) USING (experiment_id, variant, lineage_seed, generation,
                agent_id, time)
            """
    if order_results:
        sql = f"SELECT * FROM ({sql}) ORDER BY {_ID_COLS}"
    if conn is None:
        return sql
    return conn.sql(sql).pl()


def cumulative_time_history(history_sql: str) -> str:
    """Rewrite ``global_time`` to be cumulative (absolute) across generations.

    vEcoli emits a monotonic absolute ``time`` across a lineage, so a
    ``GROUP BY time`` over a multigeneration slice never merges rows from
    different generations.  v2ecoli's ``global_time`` resets to 0 each
    generation, which would make the ptools ``read_outputs`` group-by-time
    spuriously collide (and concatenate the per-row ``bulk__id`` lists).

    This adds the summed duration of all prior generations to each row's
    ``global_time``, reproducing vEcoli's absolute-time axis so the inherited
    single-scale ``analyze`` runs unchanged with an identity group-by.

    Only valid for a single-lineage slice (one cell per generation, e.g. a
    multigeneration group); do NOT use where multiple cells share a generation
    (e.g. multiseed), as that genuinely requires cross-cell aggregation.
    """
    return f"""
        WITH _gmax AS (
            SELECT generation, max(global_time) AS gmt
            FROM ({history_sql}) GROUP BY generation
        ),
        _goff AS (
            -- sum (max_time + 1s gap) of all prior generations, so each
            -- generation starts strictly after the previous one ends (no
            -- boundary-time collision between gen N's last row and gen N+1's
            -- first row, which would otherwise merge under GROUP BY time).
            SELECT generation,
                COALESCE(SUM(gmt + 1) OVER (
                    ORDER BY generation
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS off
            FROM _gmax
        )
        SELECT o.* EXCLUDE(global_time), o.global_time + g.off AS global_time
        FROM ({history_sql}) o JOIN _goff g USING (generation)
    """


def num_cells(conn: duckdb.DuckDBPyConnection, subquery: str) -> int:
    """Distinct cell count in a subquery (vEcoli parity)."""
    return conn.sql(
        f"""SELECT count(DISTINCT (experiment_id, variant, lineage_seed,
        generation, agent_id)) FROM ({subquery})"""
    ).fetchone()[0]


def skip_n_gens(subquery: str, n: int) -> str:
    """Skip the first ``n`` generations of a subquery (vEcoli parity)."""
    return f"SELECT * FROM ({subquery}) WHERE generation > {n}"


def available_columns(conn: duckdb.DuckDBPyConnection, history_sql: str) -> set[str]:
    """Return the set of column names available in ``history_sql``."""
    rows = conn.sql(
        f"SELECT column_name FROM (DESCRIBE ({history_sql}))"
    ).fetchall()
    return {r[0] for r in rows}


def bulk_field_ids(conn: duckdb.DuckDBPyConnection, history_sql: str) -> list[str]:
    """Parquet bulk-molecule ordering (equivalent of ``field_metadata("bulk")``).

    v2ecoli stores per-row ``bulk__id`` lists; the ordering is stable across
    rows/files, so the first row defines the ``bulk__count`` column order.
    """
    row = conn.sql(
        f"SELECT bulk__id FROM ({history_sql}) LIMIT 1"
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("no bulk__id rows found in history_sql")
    return list(row[0])


def bulk_count_idx_expr(
    conn: duckdb.DuckDBPyConnection,
    history_sql: str,
    mol_ids: list[str],
    names: Optional[list[str]] = None,
    zero_to_null: bool = False,
) -> str:
    """``named_idx`` expression over ``bulk__count`` for ``mol_ids``.

    Maps each molecule id to its index in the parquet bulk ordering (NOT the
    sim_data ordering) and builds ``bulk__count[idx+1] AS name`` expressions.
    Raises ``ValueError`` if any requested id is absent — never silently drops.
    """
    order = bulk_field_ids(conn, history_sql)
    pos = {bid: i for i, bid in enumerate(order)}
    missing = [m for m in mol_ids if m not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} molecule id(s) not in parquet bulk ordering: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    idxs = [pos[m] for m in mol_ids]
    if names is None:
        names = list(mol_ids)
    return named_idx("bulk__count", names, [idxs], zero_to_null=zero_to_null)


def collapse_cross_seed(
    df,
    id_cols: frozenset = frozenset({"bulk__id"}),
):
    """Collapse a multi-seed DataFrame to one row per ``time``.

    At multiseed scale, multiple seeds share the same absolute ``time``
    value.  v2ecoli's pandas ``groupby.sum()`` *concatenates* Python-list
    columns instead of adding them element-wise — this function replaces
    that groupby with a correct aggregation:

    - **id columns** (``id_cols``, e.g. ``bulk__id``): take the first
      row's value; assert every row in the group has the same length
      (raises ``ValueError`` if not — never silently truncates/pads).
    - **list/array columns** (all other non-scalar columns): element-wise
      numpy sum across seeds.  Assert all arrays in a group share the same
      shape (raises ``ValueError`` if not).
    - **scalar columns**: plain sum.

    Returns a DataFrame with one row per ``time``, columns in the original
    order.
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        return df.copy()

    first_row = df.iloc[0]

    list_cols: list[str] = []
    id_col_list: list[str] = []
    scalar_cols: list[str] = []

    for col in df.columns:
        if col == "time":
            continue
        val = first_row[col]
        if col in id_cols:
            id_col_list.append(col)
        elif isinstance(val, (list, np.ndarray)):
            list_cols.append(col)
        else:
            scalar_cols.append(col)

    result_rows: list[dict] = []
    for time_val, group in df.groupby("time", sort=True):
        row: dict = {"time": time_val}

        # id columns: take first, assert all same length
        for col in id_col_list:
            first_val = list(group[col].iloc[0])
            for other_val in group[col].iloc[1:]:
                other_list = list(other_val)
                if len(other_list) != len(first_val):
                    raise ValueError(
                        f"Mismatched length in id column '{col}' at "
                        f"time={time_val}: expected {len(first_val)}, "
                        f"got {len(other_list)}"
                    )
            row[col] = first_val

        # list/array columns: element-wise numpy sum across seeds
        for col in list_cols:
            arrays = [np.asarray(v) for v in group[col]]
            shapes = [a.shape for a in arrays]
            if len(set(shapes)) > 1:
                raise ValueError(
                    f"Mismatched array shapes in column '{col}' at "
                    f"time={time_val}: {shapes}"
                )
            row[col] = np.sum(np.stack(arrays), axis=0)

        # scalar columns: plain sum
        for col in scalar_cols:
            row[col] = group[col].sum()

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    # Preserve original column order
    return result[list(df.columns)]


def cast_decimals(df):
    """Cast any polars ``Decimal`` columns to ``Float64``.

    DuckDB surfaces some numeric columns as ``DECIMAL``; Altair's JSON encoder
    cannot serialize Python ``Decimal``, so coerce them to float before
    rendering a chart.
    """
    import polars as pl

    dec = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Decimal]
    if dec:
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in dec])
    return df


def chart_to_html(chart, title: str = "") -> str:
    """Serialize an Altair chart to a self-contained HTML view fragment.

    Altair caps inline datasets at 5000 rows by default; simulation timeseries
    routinely exceed that, so the limit is disabled (data is embedded inline in
    the view HTML, matching vEcoli's ``chart.save(path)`` behaviour).
    """
    import altair as alt

    alt.data_transformers.disable_max_rows()
    html = chart.to_html()
    if title:
        return f'<div class="analysis-view"><h3>{title}</h3>{html}</div>'
    return f'<div class="analysis-view">{html}</div>'


def ptools_heatmap_view(df: "pd.DataFrame", title: str) -> str:  # noqa: F821
    """Render a frame-ID × timepoint matrix as a heatmap HTML fragment.

    Tries Plotly ``go.Heatmap`` first (efficient for ~4500 rows × ~8 cols);
    falls back to an Altair ``mark_rect`` heatmap when Plotly is absent.
    Y-axis tick labels are suppressed (too dense at 4500 rows); frame IDs
    appear on hover.

    Args:
        df: DataFrame with frame IDs as index and timepoint strings as columns.
        title: Human-readable chart title.

    Returns:
        HTML fragment (``full_html=False`` for Plotly; ``chart.to_html()``
        div for Altair).
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Heatmap(
                z=df.values.tolist(),
                x=list(df.columns),
                y=list(df.index),
                colorbar=dict(title="Value"),
                hovertemplate=(
                    "Frame: %{y}<br>Time: %{x}<br>Value: %{z:.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            yaxis=dict(showticklabels=False),
            margin=dict(l=60, r=60, t=60, b=60),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        pass

    # --- Altair fallback ---
    import altair as alt
    import pandas as pd  # noqa: F811

    # Use a safe column name — df.index.name is "$" which is fine in Altair
    # long-form encoding but we rename to avoid any shorthand ambiguity.
    id_col = "frame_id"
    df_reset = df.copy().reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: id_col})
    long = df_reset.melt(id_vars=[id_col], var_name="timepoint", value_name="value")

    alt.data_transformers.disable_max_rows()

    chart = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X(
                "timepoint:N",
                sort=list(df.columns),
                title="Timepoint",
            ),
            y=alt.Y(
                f"{id_col}:N",
                sort=None,
                axis=alt.Axis(labels=False, ticks=False, title=None),
            ),
            color=alt.Color("value:Q", title="Value"),
            tooltip=[
                alt.Tooltip(f"{id_col}:N", title="Frame ID"),
                alt.Tooltip("timepoint:N", title="Timepoint"),
                alt.Tooltip("value:Q", title="Value", format=".4f"),
            ],
        )
        .properties(title=title, width=400, height=600)
    )

    return chart_to_html(chart, "")
