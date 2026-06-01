"""The parquet emitter must capture the raw bulk molecule counts.

Regression for the workflow parquet path: bulk (the ~16k-molecule count
array) is flattened by ParquetEmitter into parallel ``bulk__id`` /
``bulk__count`` list columns. The SQLite path deliberately omits bulk; the
parquet path includes it (see v2ecoli/composites/_helpers.py emitter step).
"""

import os

import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def test_parquet_emitter_includes_bulk_counts(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pbg_emitters")

    import glob
    from v2ecoli.workflow.run import run_workflow

    config = {
        "experiment_id": "bulkemit",
        "n_init_sims": 1,
        "generations": 1,
        "single_daughters": True,
        "cache_dir": CACHE,
        "out_dir": str(tmp_path / "parquet"),
        "variants": {},
        "max_duration_per_gen": 4.0,  # cap: 1 capped generation, fast
        "time_step": 1.0,
    }
    result = run_workflow(config, max_sim_time=20.0)
    assert result["complete"] is True

    hist = glob.glob(str(tmp_path / "parquet" / "**" / "history" / "**" / "*.pq"),
                     recursive=True)
    assert hist, "no history parquet written"

    rel = duckdb.sql(f"SELECT * FROM read_parquet('{hist[0]}')")
    cols = rel.columns
    assert "bulk__id" in cols, f"bulk__id missing; cols sample: {cols[:10]}"
    assert "bulk__count" in cols, f"bulk__count missing; cols sample: {cols[:10]}"

    ids, counts = duckdb.sql(
        f"SELECT bulk__id, bulk__count FROM read_parquet('{hist[0]}') LIMIT 1"
    ).fetchone()
    assert len(ids) == len(counts) > 1000          # full bulk vector
    assert any(c > 0 for c in counts)              # real, nonzero counts
