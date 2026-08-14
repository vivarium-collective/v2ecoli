"""Tests for the run-identity combiner (v2ecoli#472/#473).

``build_run_identity`` combines code identity + a freshly-read cache-content
fingerprint + design/grid metadata into one record; ``write_run_identity`` /
``read_run_identity`` round-trip it through the canonical ``run_identity.json``
sidecar every ``run_*`` entrypoint now writes.
"""
import json

from v2ecoli.library.cache_version import CacheVersion, write_cache_version
from v2ecoli.library.run_provenance import (
    RUN_IDENTITY_FILENAME,
    build_run_identity,
    read_run_identity,
    write_run_identity,
    write_run_identity_record,
)


def test_build_run_identity_honest_null_without_a_cache_dir():
    """Mirrors code_provenance's own honesty contract: a reason string, not a
    guess, when there's nothing to read — never a silent default."""
    record = build_run_identity(design={"experiment_id": "x"})
    assert record["cache_version"] == {
        "available": False, "reason": "no cache_dir supplied"}
    assert record["design"] == {"experiment_id": "x"}
    assert record["code"]["commit"]  # this repo's own tree is always gittable


def test_build_run_identity_honest_null_when_cache_version_missing(tmp_path):
    empty_cache_dir = tmp_path / "cache_no_version_json"
    empty_cache_dir.mkdir()
    record = build_run_identity(cache_dir=str(empty_cache_dir))
    assert record["cache_version"]["available"] is False
    assert "no" in record["cache_version"]["reason"]


def test_build_run_identity_reads_cache_version_fresh_not_by_reference(tmp_path):
    """The fingerprint must be a COPY taken at write time — cache_version.json
    is mutable and gets silently regenerated later (v2ecoli#472 §2), so a
    pointer/symlink would drift out from under the run's own record."""
    cache_dir = tmp_path / "cache"
    version = CacheVersion(schema_version="2", inputs_hash="abc123",
                           per_file_hashes={"x.py": "deadbeef"})
    write_cache_version(str(cache_dir), version=version)

    record = build_run_identity(cache_dir=str(cache_dir))
    assert record["cache_version"] == {
        "available": True, "inputs_hash": "abc123", "schema_version": "2"}

    # Mutate the shared file after the fact (simulating later regeneration);
    # a NEW build_run_identity call sees the new value — proving the first
    # call's record was a copy, not a live reference.
    write_cache_version(str(cache_dir), version=CacheVersion(
        schema_version="2", inputs_hash="changed-later", per_file_hashes={}))
    later = build_run_identity(cache_dir=str(cache_dir))
    assert later["cache_version"]["inputs_hash"] == "changed-later"
    assert record["cache_version"]["inputs_hash"] == "abc123"  # unaffected


def test_write_read_run_identity_round_trip(tmp_path):
    out_dir = tmp_path / "out" / "some_run"
    written = write_run_identity(str(out_dir), design={"experiment_id": "r1"})
    assert (out_dir / RUN_IDENTITY_FILENAME).is_file()
    assert read_run_identity(str(out_dir)) == written


def test_read_run_identity_none_when_absent(tmp_path):
    assert read_run_identity(str(tmp_path / "nothing_here")) is None


def test_read_run_identity_none_on_corrupt_json(tmp_path):
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / RUN_IDENTITY_FILENAME).write_text("{not valid json")
    assert read_run_identity(str(d)) is None


def test_write_run_identity_record_skips_recomputation(tmp_path):
    """The split write_run_identity_record(out_dir, record) exists so a caller
    that already built the record (e.g. to embed it in its own run_config
    too) doesn't pay for a second git/cache-fingerprint round trip."""
    record = {"code": {"commit": "precomputed"}, "cache_version": {}, "design": {}}
    write_run_identity_record(str(tmp_path / "d"), record)
    assert read_run_identity(str(tmp_path / "d")) == record


def test_write_run_identity_is_atomic_no_half_file_on_write(tmp_path):
    out_dir = tmp_path / "d"
    write_run_identity(str(out_dir), design={"experiment_id": "atomic"})
    # no leftover .tmp file
    assert sorted(p.name for p in out_dir.iterdir()) == [RUN_IDENTITY_FILENAME]
    with open(out_dir / RUN_IDENTITY_FILENAME) as f:
        json.load(f)  # well-formed, not a partial write


def test_write_run_identity_record_routes_an_s3_out_dir_through_save_json(
        monkeypatch, tmp_path):
    """An ``s3://`` out_dir must be delegated to ``v2ecoli.cache.save_json``,
    the same writer the sibling ``summary.json`` uses (#485).

    Regression guard with teeth: the local path is ``pathlib``-based, and
    ``Path("s3://bucket/run")`` silently collapses to ``s3:/bucket/run``. Without
    this branch a sweep dispatched to S3 writes its identity into a local
    directory literally named ``s3:`` and reads back as having none — provenance
    lost with no error, which is the failure ``run_identity.json`` exists to
    prevent. Never touches the network: ``save_json`` is monkeypatched.
    """
    import v2ecoli.cache

    calls = []
    monkeypatch.setattr(v2ecoli.cache, "save_json",
                        lambda data, path: calls.append((data, path)))
    monkeypatch.chdir(tmp_path)  # so a regression's stray local write lands here

    record = {"code": {"commit": "s3run"}, "cache_version": {}, "design": {}}
    write_run_identity_record("s3://bucket/sweeps/run-1", record)

    assert calls == [(record, f"s3://bucket/sweeps/run-1/{RUN_IDENTITY_FILENAME}")]
    # the pathlib collapse ("s3:/bucket/...") must not have happened
    assert not (tmp_path / "s3:").exists()
    assert list(tmp_path.iterdir()) == []


def test_write_run_identity_record_local_path_does_not_use_save_json(
        monkeypatch, tmp_path):
    """The other half of the branch: a local out_dir keeps the tmp-file +
    os.replace path untouched and must NOT reach for save_json."""
    import v2ecoli.cache

    def _fail(*_a, **_k):
        raise AssertionError("local write must not route through save_json")

    monkeypatch.setattr(v2ecoli.cache, "save_json", _fail)

    record = {"code": {"commit": "local"}, "cache_version": {}, "design": {}}
    write_run_identity_record(str(tmp_path / "d"), record)
    assert read_run_identity(str(tmp_path / "d")) == record
