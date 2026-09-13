"""Assemble a campaign's per-variant experiment trees into one variant-indexed store.

A *campaign* is N simulation experiments dispatched under one identity — one per design
variant (a genotype, a native design, an antibiotic dose, ...). Each lands as its own
experiment tree on S3, internally ``variant=0``; the campaign is the mapping from those
trees to their variant numbers. Historically each cross-variant analysis re-discovered that
mapping by globbing a label pattern and hand-built the union (``combine_run4_fss`` did this),
which is brittle: a superseded prefix pattern silently reads the wrong run, and a label glob
pools a good store with a defective re-fire of the same name.

:func:`assemble_campaign` takes an explicit **manifest** (one row per variant: the variant
id + its experiment prefix + provenance) and unions the trees into one variant-indexed
parquet store — ``variant`` from the manifest (overriding each tree's own ``variant=0``),
``lineage_seed`` / ``generation`` / ``global_time`` carried from the hive, ``union_by_name``
so stores with different column sets combine cleanly, provenance kept per variant, and a
**completeness assertion** so a missing or empty prefix fails loud instead of producing a
silently short store.

Domain-agnostic by construction: the caller passes which ``columns`` to carry (the
reduced-column memory win is theirs to spend) and any ``computed_columns`` (per-store SQL,
e.g. a metric reconstruction). Nothing here knows about violacein, antibiotics, or any
specific card — ``combine_run4_fss`` and the antibiotic combines become thin callers.

Cumulative ``lineage_time`` is intentionally NOT materialised here: it is a per-scale
reconstruction (:func:`v2ecoli.workflow.analyses._helpers.reconstruct_cumulative_time`) that
the analyses own. The store carries ``global_time`` + ``generation`` + ``lineage_seed``; the
caller must include whatever columns those reconstructions need in ``columns``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

from v2ecoli.library.sweep_io import (
    analysis_temp_dir,
    apply_analysis_duckdb_config,
    configure_duckdb_s3,
    history_files,
    is_s3_uri,
)

# The always-carried spine of every assembled store. ``variant`` is the manifest's
# (overriding the tree's own variant= partition); the rest carry from the hive.
_KEY_COLUMNS = ("lineage_seed", "generation", "agent_id", "global_time")

_TRANSIENT = ("HTTP", "RequestTimeTooSkewed", "Timeout", "timed out", "403",
              "IOException", "Connection", "Could not establish")


class CampaignAssemblyError(RuntimeError):
    """A campaign could not be assembled as specified — a manifest defect (duplicate
    variant) or an incompleteness (a manifest row that landed no rows). Raised instead
    of returning a silently short store."""


@dataclass
class CampaignRow:
    variant: int
    experiment_prefix: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledStore:
    """The result of :func:`assemble_campaign`."""
    path: str
    variants: list[int]
    n_rows: int
    columns: list[str]
    provenance: dict[int, dict[str, Any]]


def _parse_manifest(manifest) -> list[CampaignRow]:
    """Accept a dict ``{rows: [...]}``, a bare list of row dicts, or an already-parsed
    list of :class:`CampaignRow`. Each row needs an int ``variant`` and a non-empty
    ``experiment_prefix``; every other key is carried as provenance."""
    if isinstance(manifest, dict):
        raw = manifest.get("rows", [])
    else:
        raw = manifest
    rows: list[CampaignRow] = []
    seen: set[int] = set()
    for r in raw:
        if isinstance(r, CampaignRow):
            row = r
        else:
            if "variant" not in r or r.get("experiment_prefix") in (None, ""):
                raise CampaignAssemblyError(
                    f"manifest row missing required variant/experiment_prefix: {r!r}")
            try:
                variant = int(r["variant"])
            except (TypeError, ValueError):
                raise CampaignAssemblyError(
                    f"manifest variant must be an int, got {r['variant']!r}")
            prov = {k: v for k, v in r.items()
                    if k not in ("variant", "experiment_prefix")}
            prov["experiment_prefix"] = r["experiment_prefix"]
            row = CampaignRow(variant, str(r["experiment_prefix"]), prov)
        if row.variant in seen:
            raise CampaignAssemblyError(
                f"duplicate variant {row.variant} in manifest — ambiguous assignment")
        seen.add(row.variant)
        rows.append(row)
    if not rows:
        raise CampaignAssemblyError("manifest has no rows")
    return rows


def _sweep_dir(base: str | None, prefix: str) -> str:
    if not base:
        return prefix
    return base.rstrip("/") + "/" + prefix.strip("/")


def _store_columns(conn, files: list[str]) -> set[str]:
    flist = "[" + ",".join(f"'{f}'" for f in files) + "]"
    rows = conn.sql(
        f"DESCRIBE SELECT * FROM read_parquet({flist}, hive_partitioning=true, "
        f"union_by_name=true)").fetchall()
    return {r[0] for r in rows}


def _row_select(files: list[str], variant: int, present: list[str],
                computed_columns: dict[str, str] | None, gen_lb: int) -> str:
    flist = "[" + ",".join(f"'{f}'" for f in files) + "]"
    proj = [f"CAST({variant} AS BIGINT) AS variant"]
    for k in _KEY_COLUMNS:
        proj.append(f"CAST({k} AS BIGINT) AS {k}" if k in ("lineage_seed", "generation")
                    else k)
    proj.extend(present)
    for name, expr in (computed_columns or {}).items():
        proj.append(f"({expr}) AS {name}")
    return (
        f"SELECT {', '.join(proj)} "
        f"FROM read_parquet({flist}, hive_partitioning=true, union_by_name=true) "
        f"WHERE generation >= {int(gen_lb)}"
    )


def _connect(base: str | None, mem_limit: str | None):
    conn = duckdb.connect()
    if base and is_s3_uri(base):
        configure_duckdb_s3(conn)
    # Spill to a real temp dir at the container's budget (or an explicit mem_limit),
    # single-threaded — the #789/#795 memory lessons for wide reads.
    conn.execute(f"SET temp_directory = '{analysis_temp_dir()}'")
    conn.execute("SET preserve_insertion_order = false")
    if mem_limit:
        conn.execute(f"SET memory_limit = '{mem_limit}'")
    else:
        apply_analysis_duckdb_config(conn, threads=1)
    conn.execute("SET threads = 1")
    for stmt in ("SET http_timeout = 120000", "SET http_retries = 6"):
        try:
            conn.execute(stmt)
        except duckdb.Error:
            pass
    return conn


def _copy_with_retry(base, mem_limit, select_sql, out_path, tag="", tries=5):
    """COPY one variant's reduced rows to a local parquet, retrying transient S3 errors
    (RequestTimeTooSkewed / 403 / stalled reads) with a fresh connection each attempt."""
    tmp = out_path + ".tmp"
    for attempt in range(1, tries + 1):
        conn = _connect(base, mem_limit)
        try:
            conn.execute(f"COPY ({select_sql}) TO '{tmp}' (FORMAT parquet)")
            os.replace(tmp, out_path)
            return
        except (duckdb.HTTPException, duckdb.IOException) as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == tries or not any(s in str(e) for s in _TRANSIENT):
                raise
            time.sleep(3 * attempt)
        finally:
            conn.close()


def assemble_campaign(manifest, *, columns, out_path, base=None,
                      generation_lower_bound=0, computed_columns=None,
                      mem_limit=None, parts_dir=None) -> AssembledStore:
    """Union a campaign's per-variant experiment trees into one variant-indexed store.

    Parameters
    ----------
    manifest : dict | list
        ``{"rows": [{"variant": int, "experiment_prefix": str, ...provenance}]}`` (or a
        bare list of such rows, or ``CampaignRow`` objects). ``variant`` is required and
        explicit — it overrides each tree's own ``variant=`` partition. Duplicate
        variants are rejected.
    columns : list[str]
        Parquet columns to carry beyond the key spine (variant/lineage_seed/generation/
        agent_id/global_time). Per store only the columns actually present are read;
        the final union is ``union_by_name`` so a column present in one store and absent
        in another becomes null rather than a failed read. This is the reduced-column
        memory win — it is the caller's to choose.
    out_path : str
        Local path for the assembled parquet.
    base : str, optional
        Location each row's ``experiment_prefix`` resolves under — a local directory (tests)
        or an object-store root such as ``s3://<bucket>/vecoli-output``. When omitted, each
        ``experiment_prefix`` is treated as a complete path/URI.
    generation_lower_bound : int
        Drop generations below this (e.g. skip pre-steady-state generations).
    computed_columns : dict[str, str], optional
        ``{name: sql_expr}`` extra columns added per store — the seam a domain caller uses
        for a metric reconstruction (e.g. splicing a value into a reconstructed list). The
        expression must reference only columns present in every store it applies to.
    mem_limit : str, optional
        Explicit DuckDB ``memory_limit`` (e.g. ``"40GB"``); otherwise the container budget.
    parts_dir : str, optional
        Where per-variant staged parts are written (default ``<out_path>.parts``).

    Returns
    -------
    AssembledStore
        Path + the variants, row count, final column list, and per-variant provenance.

    Raises
    ------
    CampaignAssemblyError
        On a manifest defect (duplicate variant) or incompleteness — any manifest row
        that resolves to no history parquet or lands zero rows fails loud, so a dropped
        prefix never yields a silently short store.
    """
    rows = _parse_manifest(manifest)
    stage = parts_dir or (out_path + ".parts")
    os.makedirs(stage, exist_ok=True)

    describe_conn = _connect(base, mem_limit)
    parts: list[str] = []
    missing: list[str] = []
    provenance: dict[int, dict] = {}
    try:
        for row in rows:
            provenance[row.variant] = dict(row.provenance)
            sweep = _sweep_dir(base, row.experiment_prefix)
            files = history_files(sweep)
            if not files:
                missing.append(f"variant {row.variant} ({row.experiment_prefix}): "
                               "no history parquet")
                continue
            present = [c for c in columns if c in _store_columns(describe_conn, files)]
            part = os.path.join(stage, f"variant_{row.variant}.parquet")
            _copy_with_retry(
                base, mem_limit,
                _row_select(files, row.variant, present, computed_columns,
                            generation_lower_bound),
                part, tag=f"variant{row.variant}")
            n = describe_conn.sql(
                f"SELECT count(*) FROM read_parquet('{part}')").fetchone()[0]
            if n == 0:
                missing.append(f"variant {row.variant} ({row.experiment_prefix}): "
                               f"0 rows after generation >= {generation_lower_bound}")
                continue
            parts.append(part)
    finally:
        describe_conn.close()

    if missing:
        raise CampaignAssemblyError(
            "campaign incomplete — these manifest rows landed no rows (a dropped/empty "
            "prefix, NOT a silently short store):\n  " + "\n  ".join(missing))

    union_conn = _connect(base, mem_limit)
    try:
        flist = "[" + ",".join(f"'{p}'" for p in parts) + "]"
        union_conn.execute(
            f"COPY (SELECT * FROM read_parquet({flist}, union_by_name=true)) "
            f"TO '{out_path}' (FORMAT parquet)")
        n_rows, n_variants = union_conn.sql(
            f"SELECT count(*), count(DISTINCT variant) FROM read_parquet('{out_path}')"
        ).fetchone()
        cols = [r[0] for r in union_conn.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{out_path}')").fetchall()]
    finally:
        union_conn.close()

    # Second completeness gate: the assembled store must carry every manifest variant.
    if n_variants != len(rows):
        got = set()
        c2 = duckdb.connect()
        try:
            got = {r[0] for r in c2.sql(
                f"SELECT DISTINCT variant FROM read_parquet('{out_path}')").fetchall()}
        finally:
            c2.close()
        lost = sorted(set(r.variant for r in rows) - got)
        raise CampaignAssemblyError(
            f"assembled store has {n_variants} variants, manifest has {len(rows)}; "
            f"missing {lost}")

    return AssembledStore(
        path=out_path,
        variants=sorted(r.variant for r in rows),
        n_rows=int(n_rows),
        columns=cols,
        provenance=provenance,
    )
