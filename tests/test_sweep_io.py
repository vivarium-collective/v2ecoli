"""Sweep location/access: one file list whether the sweep is local or on S3.

The S3 branch is exercised with a fake DuckDB connection rather than a live
bucket, so these run without credentials or network. What matters and is
asserted: the glob pattern handed to object storage matches the local glob's
shape, and a local sweep never reaches for httpfs.
"""
from __future__ import annotations

import pytest

from v2ecoli.library import sweep_io


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture()
def local_sweep(tmp_path):
    """A sweep laid out the way the emitter writes it, plus decoy files."""
    root = tmp_path / "sweep"
    hist = root / "history" / "experiment_id=e" / "variant=0"
    _touch(hist / "lineage_seed=0" / "generation=0" / "agent_id=0" / "0.pq")
    _touch(hist / "lineage_seed=1" / "generation=3" / "agent_id=01" / "800.pq")
    # decoys: right suffix, wrong subtree / wrong suffix in the right subtree
    _touch(root / "success" / "experiment_id=e" / "s.pq")
    _touch(hist / "lineage_seed=0" / "generation=0" / "agent_id=0" / "notes.txt")
    return root


def test_is_s3_uri_discriminates():
    assert sweep_io.is_s3_uri("s3://bucket/sweep")
    assert not sweep_io.is_s3_uri("/local/sweep")
    assert not sweep_io.is_s3_uri("relative/sweep")


def test_history_files_finds_only_history_parquet(local_sweep):
    files = sweep_io.history_files(str(local_sweep))
    assert len(files) == 2
    assert all(f.endswith(".pq") for f in files)
    assert all("history" in f for f in files)
    assert not any("success" in f for f in files)


def test_history_files_excludes_stray_flat_default_tree(local_sweep):
    """A stray non-hive ``default/history/1.pq`` (an emit that fell back to
    experiment_id "default" and wrote a flat file with no partition keys) must
    NOT be returned — mixing it into ``read_parquet(hive_partitioning=true)``
    aborts the whole read with "Hive partition mismatch". Only files under a
    ``history/experiment_id=…`` segment are real hive history.
    """
    _touch(local_sweep / "default" / "history" / "1.pq")
    files = sweep_io.history_files(str(local_sweep))
    assert len(files) == 2
    assert all("experiment_id=" in f for f in files)
    assert not any(f.endswith("/default/history/1.pq") for f in files)


def test_history_files_is_sorted_and_stable(local_sweep):
    files = sweep_io.history_files(str(local_sweep))
    assert files == sorted(files)
    assert files == sweep_io.history_files(str(local_sweep))


def test_history_files_missing_dir_is_empty_not_error(tmp_path):
    assert sweep_io.history_files(str(tmp_path / "absent")) == []


class _FakeConn:
    """Records the SQL it is asked to run; returns one object row."""

    def __init__(self):
        self.sql_seen, self.exec_seen, self.closed = [], [], False

    def sql(self, q):
        self.sql_seen.append(q)
        return self

    def execute(self, q):
        self.exec_seen.append(q)

    def fetchall(self):
        return [("s3://bucket/sweep/history/experiment_id=e/generation=0/x.pq",)]

    def close(self):
        self.closed = True


def test_history_files_s3_globs_object_storage(monkeypatch):
    """The S3 branch lists via DuckDB's glob() using the local glob's shape."""
    import duckdb

    conn = _FakeConn()
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: conn)
    monkeypatch.setattr(sweep_io, "configure_duckdb_s3", lambda c, region=None: None)

    files = sweep_io.history_files("s3://bucket/sweep/")

    assert files == ["s3://bucket/sweep/history/experiment_id=e/generation=0/x.pq"]
    # trailing slash normalized; hive-scoped shape matches the local glob so a
    # stray flat default/history/*.pq is excluded on S3 too
    assert (
        "glob('s3://bucket/sweep/**/history/experiment_id=*/**/*.pq')"
        in conn.sql_seen[0]
    )
    assert conn.closed, "the listing connection should not be leaked"


def test_connect_for_local_does_not_touch_httpfs(monkeypatch, local_sweep):
    """A local sweep must not require credentials or the httpfs extension."""
    called = []
    monkeypatch.setattr(sweep_io, "configure_duckdb_s3",
                        lambda c, region=None: called.append(c))

    conn = sweep_io.connect_for(str(local_sweep))
    try:
        assert called == []
        assert conn.sql("SELECT 42").fetchall() == [(42,)]
    finally:
        conn.close()


def test_connect_for_s3_configures_credentials(monkeypatch):
    import duckdb

    conn = _FakeConn()
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: conn)
    called = []
    monkeypatch.setattr(sweep_io, "configure_duckdb_s3",
                        lambda c, region=None: called.append(c))

    assert sweep_io.connect_for("s3://bucket/sweep") is conn
    assert called == [conn]


def test_analysis_runner_reexports_the_same_objects():
    """#418's callers import these from the runner; keep that surface intact."""
    from v2ecoli.workflow import analysis_runner as ar

    assert ar.is_s3_uri is sweep_io.is_s3_uri
    assert ar.history_files is sweep_io.history_files
    assert ar.configure_duckdb_s3 is sweep_io.configure_duckdb_s3
    assert ar._S3_PREFIX == sweep_io._S3_PREFIX
