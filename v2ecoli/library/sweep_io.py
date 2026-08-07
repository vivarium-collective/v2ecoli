"""Locating and reading a sweep's parquet, wherever the sweep lives.

A ``sweep_dir`` may be a local path OR an ``s3://`` URI. Both forms resolve to
the same flat list of history parquet files, so callers do not branch on
storage: :func:`history_files` hands back paths DuckDB can read either way, and
:func:`configure_duckdb_s3` gives a connection the credentials to do it.

These were introduced in the analysis runner (#418) for the standalone
``v2ecoli-analyze <sweep>`` path. They live here because the report-card vector
extraction (:mod:`v2ecoli.library.card_vectors`) needs them too, and
``library`` does not import from ``workflow``. ``workflow.analysis_runner``
re-exports them, so its existing callers are unaffected.
"""
from __future__ import annotations

import glob
import os

_S3_PREFIX = "s3://"


def is_s3_uri(path: str) -> bool:
    return str(path).startswith(_S3_PREFIX)


def history_files(sweep_dir: str) -> list[str]:
    """The sweep's history parquet, as local paths or ``s3://`` URIs.

    Mirrors the local glob for S3 so every caller (FROM-clause builder, cell-key
    enumeration, per-cell record builder) sees one file list regardless of where
    the sweep lives.
    """
    if not is_s3_uri(sweep_dir):
        return sorted(glob.glob(
            os.path.join(sweep_dir, "**", "history", "**", "*.pq"), recursive=True))
    # DuckDB's glob() lists object storage through the same httpfs extension the
    # read needs — so listing costs no extra dependency and no parquet read.
    import duckdb

    conn = duckdb.connect()
    configure_duckdb_s3(conn)
    pattern = sweep_dir.rstrip("/") + "/**/history/**/*.pq"
    try:
        rows = conn.sql(
            f"SELECT file FROM glob('{pattern}')").fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)


def configure_duckdb_s3(conn, region: str | None = None) -> None:
    """Give a DuckDB connection S3 read access via httpfs.

    Credentials come from the standard AWS chain. SSO callers should export them
    first (``eval "$(aws configure export-credentials --format env)"``); on an
    EC2/Batch node the instance role resolves through boto3 the same way.
    """
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    region = region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
        "AWS_REGION") or "us-gov-west-1"
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    token = os.environ.get("AWS_SESSION_TOKEN")
    if not (key and secret):
        import boto3

        creds = boto3.Session().get_credentials()
        if creds is None:
            raise RuntimeError(
                "no AWS credentials for the S3 sweep — export them with "
                '`eval "$(aws configure export-credentials --format env)"`')
        frozen = creds.get_frozen_credentials()
        key, secret, token = frozen.access_key, frozen.secret_key, frozen.token
    parts = [f"KEY_ID '{key}'", f"SECRET '{secret}'", f"REGION '{region}'"]
    if token:
        parts.append(f"SESSION_TOKEN '{token}'")
    conn.execute(f"CREATE OR REPLACE SECRET v2e_sweep_s3 (TYPE s3, {', '.join(parts)});")


def connect_for(sweep_dir: str):
    """A DuckDB connection ready to read ``sweep_dir`` (S3-configured if needed).

    The counterpart to :func:`history_files` for callers that own their own
    connection: pair the two and neither has to know where the sweep lives.
    """
    import duckdb

    conn = duckdb.connect()
    if is_s3_uri(sweep_dir):
        configure_duckdb_s3(conn)
    return conn
