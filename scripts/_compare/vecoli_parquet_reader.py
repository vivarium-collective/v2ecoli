"""Read comparison observables from vEcoli's Nextflow ParquetEmitter output on S3.

vEcoli's Nextflow workflow writes per-timestep history as Hive-partitioned
parquet under::

    <experiment>/cond_<media>/history/experiment_id=.../variant=.../
        lineage_seed=<N>/generation=<G>/agent_id=<id>/<tick>.pq

Each ``<tick>.pq`` is one partition of a single timeseries table per
(lineage_seed, generation): rows are timesteps, columns are the emitted paths
with the nested ``listeners.mass.cell_mass`` dotted path FLATTENED to a
double-underscore column name ``listeners__mass__cell_mass``.

``read_vecoli_finals`` reads the history parquet for an experiment, and for each
lineage_seed returns the FINAL (latest-time, across all generations) value of
each requested comparison observable — the same ``{seed: {obs: value}}`` shape
the zarr reader in ``s3_compare_report.py`` produces for the v2ecoli engine, so
the two pair seed-by-seed straight into ``multiseed_report._stats``/``_verdict``.

DuckDB reads the parquet directly off S3 (column projection means only the
needed observable columns are fetched, not the full ~230-column rows). The file
list is resolved with s3fs and handed to DuckDB explicitly; DuckDB still parses
the Hive partition columns (lineage_seed, generation) from each path.

Credentials: both s3fs and DuckDB use the standard AWS chain. Export SSO creds
into the environment first (the aiobotocore SSO-token refresh is unreliable)::

    export AWS_PROFILE=stanford-sso AWS_DEFAULT_REGION=us-gov-west-1
    eval "$(aws configure export-credentials --format env)"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_comparison_ensemble import COMPARISON_PATHS

# leaf name (e.g. "cell_mass") and dotted path -> flattened parquet column.
# Leaves are unique within COMPARISON_PATHS, so the leaf->column map is
# unambiguous; the dotted form is also accepted directly.
_LEAF_TO_COL = {p.split(".")[-1]: p.replace(".", "__") for p in COMPARISON_PATHS}
_DOTTED_TO_COL = {p: p.replace(".", "__") for p in COMPARISON_PATHS}


def _observable_to_column(obs: str) -> str:
    """Map a comparison observable identifier to its parquet column name.

    Accepts a leaf name ("cell_mass"), a dotted path
    ("listeners.mass.cell_mass"), or an already-flattened column
    ("listeners__mass__cell_mass").
    """
    if obs in _LEAF_TO_COL:
        return _LEAF_TO_COL[obs]
    if obs in _DOTTED_TO_COL:
        return _DOTTED_TO_COL[obs]
    if "." in obs:
        return obs.replace(".", "__")
    return obs  # assume already a flattened column


def _s3fs(region: str):
    import s3fs
    return s3fs.S3FileSystem(client_kwargs={"region_name": region})


def _list_history_files(fs, bucket: str, experiment_prefix: str) -> list[str]:
    """Resolve the history *.pq keys for an experiment (exact / prefix / substring).

    ``experiment_prefix`` is a key path under the bucket (the leading
    ``s3://<bucket>/`` is optional) and may be an exact experiment dir, a path
    prefix, or carry a substring in its final segment — mirroring the
    substring matching the zarr side uses (sms-api wraps the submitted id as
    ``sim<ID>-<experiment>-<hash>``).
    """
    ep = experiment_prefix
    if ep.startswith("s3://"):
        ep = ep[len("s3://"):]
    if ep.startswith(bucket + "/"):
        ep = ep[len(bucket) + 1:]
    ep = ep.strip("/")
    root, _, tail = ep.rpartition("/")
    globs = [
        f"{bucket}/{ep}/**/history/**/*.pq",            # exact experiment dir
        f"{bucket}/{ep}*/**/history/**/*.pq",           # prefix match
    ]
    if root:                                            # *substring* in last segment
        globs.append(f"{bucket}/{root}/*{tail}*/**/history/**/*.pq")
    for g in globs:
        hits = fs.glob(g)
        if hits:
            return sorted(hits)
    return []


def _create_s3_secret(con, region: str) -> None:
    """Configure DuckDB's httpfs S3 access from env creds (or the boto3 chain)."""
    con.execute("INSTALL httpfs; LOAD httpfs;")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    token = os.environ.get("AWS_SESSION_TOKEN")
    if not (key and secret):
        import boto3
        cr = boto3.Session().get_credentials()
        if cr is None:
            raise RuntimeError(
                "no AWS credentials — export them: "
                "`eval \"$(aws configure export-credentials --format env)\"`")
        fc = cr.get_frozen_credentials()
        key, secret, token = fc.access_key, fc.secret_key, fc.token
    parts = [f"KEY_ID '{key}'", f"SECRET '{secret}'", f"REGION '{region}'"]
    if token:
        parts.append(f"SESSION_TOKEN '{token}'")
    con.execute(f"CREATE OR REPLACE SECRET ve_s3 (TYPE s3, {', '.join(parts)});")


def read_vecoli_finals(experiment_prefix: str,
                       bucket: str,
                       observables,
                       region: str = "us-gov-west-1",
                       summary: str = "final") -> dict[int, dict[str, float]]:
    """vEcoli per-seed final observable values from S3 parquet history.

    Parameters
    ----------
    experiment_prefix : key path under ``bucket`` to the vEcoli experiment
        (e.g. ``"vecoli-output/sim47-ve-nf5-basal-1132"``); a substring in the
        final path segment is matched like the zarr side.
    bucket : S3 bucket name.
    observables : iterable of comparison observable identifiers (leaf names such
        as ``"cell_mass"``, dotted paths, or flattened columns). Observables with
        no corresponding parquet column are skipped.
    region : AWS region for the bucket.
    summary : ``"final"`` (default) = latest-time value across all generations;
        ``"mean-last-gen"`` = mean over all ticks of the last generation.

    Returns
    -------
    ``{lineage_seed: {observable: float}}`` keyed by the *input* observable
    identifiers (so it pairs directly with the v2ecoli zarr finals).
    """
    import duckdb

    observables = list(observables)
    fs = _s3fs(region)
    files = _list_history_files(fs, bucket, experiment_prefix)
    if not files:
        raise FileNotFoundError(
            f"no history *.pq found for experiment_prefix={experiment_prefix!r} "
            f"in bucket {bucket!r}")
    uris = ["s3://" + f for f in files]

    con = duckdb.connect()
    _create_s3_secret(con, region)

    # Which requested observables actually exist as columns in the data?
    present_cols = {r[0] for r in
                    con.execute(f"DESCRIBE SELECT * FROM read_parquet('{uris[0]}')").fetchall()}
    obs_to_col = {}
    for obs in observables:
        col = _observable_to_column(obs)
        if col in present_cols:
            obs_to_col[obs] = col
    if not obs_to_col:
        return {}

    file_list = ", ".join(f"'{u}'" for u in uris)
    # Quote columns; alias each back to its observable identifier.
    sel = ", ".join(f'"{c}" AS "{obs}"' for obs, c in obs_to_col.items())
    aliases = ", ".join(f'"{obs}"' for obs in obs_to_col)

    if summary == "mean-last-gen":
        # mean over all ticks of the max generation per lineage_seed
        agg = ", ".join(f'avg("{obs}") AS "{obs}"' for obs in obs_to_col)
        query = f"""
            WITH t AS (
                SELECT lineage_seed, generation, {sel}
                FROM read_parquet([{file_list}], hive_partitioning=true)
            ),
            mg AS (SELECT lineage_seed, max(generation) AS g FROM t GROUP BY lineage_seed)
            SELECT t.lineage_seed, {agg}
            FROM t JOIN mg ON t.lineage_seed = mg.lineage_seed AND t.generation = mg.g
            GROUP BY t.lineage_seed
            ORDER BY t.lineage_seed
        """
    else:  # "final" — latest (generation, time) row per lineage_seed
        query = f"""
            WITH t AS (
                SELECT lineage_seed, generation, time, {sel}
                FROM read_parquet([{file_list}], hive_partitioning=true)
            ),
            ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY lineage_seed ORDER BY generation DESC, time DESC
                ) AS rn FROM t
            )
            SELECT lineage_seed, {aliases} FROM ranked WHERE rn = 1
            ORDER BY lineage_seed
        """

    cur = con.execute(query)
    col_names = [d[0] for d in cur.description]
    out: dict[int, dict[str, float]] = {}
    for row in cur.fetchall():
        rec = dict(zip(col_names, row))
        seed = int(rec.pop("lineage_seed"))
        out[seed] = {k: (float(v) if v is not None else None) for k, v in rec.items()}
    return out


def read_vecoli_trajectory(experiment_prefix: str,
                           bucket: str,
                           observables,
                           lineage_seed: int = 0,
                           region: str = "us-gov-west-1") -> dict[str, tuple]:
    """vEcoli per-tick TRAJECTORY (value vs global time) for one lineage_seed.

    Reads every history tick for the given ``lineage_seed`` across all
    generations and returns, per requested observable, the time-ordered
    ``(times, values)`` arrays. The parquet ``time`` column is already
    global-continuous across generations (gen2 picks up where gen1 left off),
    so the arrays span the whole lineage on one seconds axis.

    Returns
    -------
    ``{observable: (numpy times[], numpy values[])}`` keyed by the *input*
    observable identifiers. Observables absent from the parquet are skipped.
    Also includes ``"_generation"`` -> (times, gen-number) for gen-boundary use.
    """
    import duckdb
    import numpy as np

    observables = list(observables)
    fs = _s3fs(region)
    files = _list_history_files(fs, bucket, experiment_prefix)
    files = [f for f in files if f"lineage_seed={lineage_seed}/" in f]
    if not files:
        raise FileNotFoundError(
            f"no history *.pq for experiment_prefix={experiment_prefix!r} "
            f"lineage_seed={lineage_seed} in bucket {bucket!r}")
    uris = ["s3://" + f for f in files]

    con = duckdb.connect()
    _create_s3_secret(con, region)
    present_cols = {r[0] for r in
                    con.execute(f"DESCRIBE SELECT * FROM read_parquet('{uris[0]}')").fetchall()}
    obs_to_col = {}
    for obs in observables:
        col = _observable_to_column(obs)
        if col in present_cols:
            obs_to_col[obs] = col
    if not obs_to_col:
        return {}

    file_list = ", ".join(f"'{u}'" for u in uris)
    sel = ", ".join(f'"{c}" AS "{obs}"' for obs, c in obs_to_col.items())
    query = f"""
        SELECT time, generation, {sel}
        FROM read_parquet([{file_list}], hive_partitioning=true)
        ORDER BY time
    """
    cur = con.execute(query)
    col_names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    arr = {c: np.array([r[i] for r in rows], dtype=float)
           for i, c in enumerate(col_names)}
    out: dict[str, tuple] = {}
    t = arr["time"]
    for obs in obs_to_col:
        out[obs] = (t, arr[obs])
    out["_generation"] = (t, arr["generation"])
    return out


if __name__ == "__main__":  # quick manual check against live data
    import argparse
    import json

    p = argparse.ArgumentParser(description="print per-seed vEcoli parquet finals")
    p.add_argument("--bucket",
                   default="smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91")
    p.add_argument("--experiment-prefix",
                   default="vecoli-output/sim47-ve-nf5-basal-1132")
    p.add_argument("--region", default="us-gov-west-1")
    p.add_argument("--summary", choices=("final", "mean-last-gen"), default="final")
    args = p.parse_args()

    obs = [p.split(".")[-1] for p in COMPARISON_PATHS]
    finals = read_vecoli_finals(args.experiment_prefix, args.bucket, obs,
                                region=args.region, summary=args.summary)
    print(json.dumps(finals, indent=2))
