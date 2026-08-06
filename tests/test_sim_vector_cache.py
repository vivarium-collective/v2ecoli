"""Tests for the run-keyed sim-vector cache.

The real extraction scans a whole sweep (~8 min over 1052 parquet files), so
these stub ``extract_vectors`` and exercise the cache's own contract: the key,
the location policy, hit-vs-miss, and the provenance it stamps.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from v2ecoli.library import sim_vector_cache as svc
from v2ecoli.library.card_vectors import EXTRACTOR_VERSION

_FAKE = {"omics": {"proteome": {"vector": [1.0, 2.0], "n_cells": 7},
                   "transcriptome": {"vector": [3.0], "n_cells": 7}},
         "fluxes": {"exchange": {"vector": [0.5], "n_cells": 7,
                                 "per_cell": [[0.5]]}}}


@pytest.fixture
def sweep(tmp_path):
    """A sweep dir carrying the authoritative experiment_id partition segment."""
    d = tmp_path / "out" / "demo_run"
    (d / "history" / "experiment_id=demo_run").mkdir(parents=True)
    return d


@pytest.fixture
def counting_extract(monkeypatch):
    """Stub extract_vectors, counting calls so hit/miss is observable."""
    calls = []

    def _fake(sweep_dir, generation_lower_bound=0):
        calls.append((sweep_dir, generation_lower_bound))
        return _FAKE

    monkeypatch.setattr(svc, "extract_vectors", _fake)
    return calls


def test_key_uses_experiment_id_gen_lb_and_extractor_version(sweep):
    p = svc.cache_path(str(sweep), generation_lower_bound=3)
    assert p.name == f"demo_run__gen3__v{EXTRACTOR_VERSION}.json"
    assert p.parent.name == "sim_vectors"
    # experiment_id comes from the partition segment, not the dir basename
    assert "demo_run" in p.name


def test_gen_lb_changes_the_key(sweep):
    """gen_lb changes the ensemble, so it must change the key — a cache keyed
    on the run alone would serve a stale vector after a burn-in change."""
    assert svc.cache_path(str(sweep), 3) != svc.cache_path(str(sweep), 0)


def test_miss_then_hit_does_not_re_extract(sweep, counting_extract):
    first = svc.load_or_extract(str(sweep), generation_lower_bound=3)
    second = svc.load_or_extract(str(sweep), generation_lower_bound=3)
    assert len(counting_extract) == 1, "second call re-extracted; cache did not hit"
    assert first["vectors"] == second["vectors"] == _FAKE


def test_refresh_forces_re_extraction(sweep, counting_extract):
    svc.load_or_extract(str(sweep), generation_lower_bound=3)
    svc.load_or_extract(str(sweep), generation_lower_bound=3, refresh=True)
    assert len(counting_extract) == 2


def test_envelope_records_run_identity_and_two_commits(sweep, counting_extract):
    env = svc.load_or_extract(str(sweep), generation_lower_bound=3)
    assert env["schema"] == svc.CACHE_SCHEMA
    assert env["run"]["experiment_id"] == "demo_run"
    assert env["run"]["generation_lower_bound"] == 3
    assert env["extractor"]["version"] == EXTRACTOR_VERSION
    prov = env["provenance"]
    # Two distinct commits, never one ambiguous `sim_commit`.
    assert "run_commit" in prov and "extracted_at_commit" in prov
    assert "sim_commit" not in prov
    # The extracting tree is knowable, so this one is non-null.
    assert prov["extracted_at_commit"]["commit"]


def test_run_commit_is_null_when_the_sweep_recorded_none(sweep, counting_extract):
    """Honest null beats a plausible-looking substitute: stamping the extracting
    tree's HEAD as the run's commit would assert something false."""
    env = svc.load_or_extract(str(sweep), generation_lower_bound=3)
    assert env["provenance"]["run_commit"] is None


def test_run_commit_is_read_when_the_sweep_recorded_one(sweep, counting_extract):
    (sweep / "run_config.json").write_text(json.dumps({"commit": "abc123def"}))
    env = svc.load_or_extract(str(sweep), generation_lower_bound=3)
    assert env["provenance"]["run_commit"] == "abc123def"


def test_out_dir_redirects_the_cache(sweep, counting_extract, tmp_path):
    out = tmp_path / "elsewhere"
    svc.load_or_extract(str(sweep), 3, out_dir=str(out))
    assert (out / "sim_vectors").is_dir()
    assert not (sweep / "sim_vectors").exists()


def test_s3_sweep_requires_an_out_dir():
    """An s3:// sweep is read-only, so the cache cannot land beside it."""
    with pytest.raises(ValueError, match="out_dir is required"):
        svc.cache_path("s3://bucket/sweep", 3)
    p = svc.cache_path("s3://bucket/sweep", 3, out_dir="/tmp/o")
    assert p.name.startswith("sweep__gen3__")


def test_cached_vectors_is_a_drop_in_for_extract_vectors(sweep, counting_extract):
    assert svc.cached_vectors(str(sweep), 3) == _FAKE


def test_stale_schema_is_re_extracted(sweep, counting_extract):
    svc.load_or_extract(str(sweep), 3)
    p = svc.cache_path(str(sweep), 3)
    blob = json.load(open(p))
    blob["schema"] = "sim_vector_cache/v0"
    p.write_text(json.dumps(blob))
    svc.load_or_extract(str(sweep), 3)
    assert len(counting_extract) == 2, "an unrecognized schema must not be trusted"
