"""Unit tests for the domain-agnostic campaign assembler (workflow/campaign_assembly.py).

assemble_campaign(manifest) unions per-variant experiment trees into one variant-indexed
parquet store. These tests run entirely on SYNTHETIC LOCAL hive parquet (no S3), so they
are CI-runnable; the equivalence-vs-combine_run4_fss check on real GovCloud stores is a
separate, S3-gated test.

They pin the behaviours B4 exists to guarantee:
* variant comes from the MANIFEST (explicit), overriding each store's own variant= partition
  (every per-genotype store is internally variant=0; the campaign says which variant it is);
* lineage_seed / generation / global_time carry through from the hive;
* union_by_name across stores with DIFFERENT column sets (a heterologous column present in
  one arm, absent in another) → nulls, not a failed read;
* generation_lower_bound filters;
* computed_columns (the caller's per-store SQL, e.g. a metric splice) are added;
* provenance is carried per variant in the report;
* COMPLETENESS is asserted — a manifest row that lands zero rows FAILS LOUD (the missing/
  short-prefix bug class B4 kills), rather than silently producing a short store.
"""

from __future__ import annotations

import os

import duckdb
import pytest

from v2ecoli.workflow.campaign_assembly import (
    assemble_campaign,
    CampaignAssemblyError,
)


def _write_hive_cell(root, prefix, *, seed, generation, cols):
    """Write one synthetic history .pq at the real sim-store hive path
    <root>/<prefix>/batch_baseline/<prefix>/history/experiment_id=<prefix>/
    variant=0/lineage_seed=<seed>/generation=<gen>/agent_id=<a>/data.pq
    (variant=0 internally — the manifest overrides it). ``cols`` maps column name
    -> scalar value; a single row is written."""
    agent = "0" * (generation + 1)
    d = os.path.join(root, prefix, "batch_baseline", prefix, "history",
                     f"experiment_id={prefix}", "variant=0",
                     f"lineage_seed={seed}", f"generation={generation}",
                     f"agent_id={agent}")
    os.makedirs(d, exist_ok=True)
    con = duckdb.connect()
    sel = ", ".join(f"{v} AS {k}" for k, v in cols.items())
    con.execute(f"COPY (SELECT {sel}) TO '{os.path.join(d, 'data.pq')}' (FORMAT parquet)")
    con.close()


def _make_two_store_campaign(root):
    """Two stores: expA (2 seeds x 2 gens, columns global_time+x) and expB (1 seed x
    2 gens, columns global_time+x+y — y present only in B). Manifest maps them to
    variants 10 and 20."""
    for seed in (0, 1):
        for g in (0, 1):
            _write_hive_cell(root, "expA", seed=seed, generation=g,
                             cols={"global_time": float(g), "x": float(10 * seed + g)})
    for g in (0, 1):
        _write_hive_cell(root, "expB", seed=0, generation=g,
                         cols={"global_time": float(g), "x": float(100 + g),
                               "y": float(g)})
    manifest = {
        "campaign_id": "unit-test",
        "rows": [
            {"variant": 10, "experiment_prefix": "expA", "db_id": 1, "arm": "left"},
            {"variant": 20, "experiment_prefix": "expB", "db_id": 2, "arm": "right"},
        ],
    }
    return manifest


def _read(path):
    con = duckdb.connect()
    df = con.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY variant, lineage_seed, "
                 f"generation").df()
    con.close()
    return df


def test_variant_from_manifest_overrides_store_partition(tmp_path):
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    out = os.path.join(root, "campaign.parquet")
    store = assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root)
    df = _read(out)
    # Both stores are internally variant=0; the manifest's 10/20 must win.
    assert sorted(df["variant"].unique().tolist()) == [10, 20]
    assert store.variants == [10, 20]
    # expA has 4 cells (2 seeds x 2 gens), expB has 2 (1 seed x 2 gens).
    assert (df["variant"] == 10).sum() == 4
    assert (df["variant"] == 20).sum() == 2
    # key columns are integer-typed
    for c in ("variant", "lineage_seed", "generation"):
        assert str(df[c].dtype).startswith(("int", "Int")), (c, df[c].dtype)


def test_union_by_name_fills_absent_columns(tmp_path):
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    out = os.path.join(root, "campaign.parquet")
    assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root)
    df = _read(out)
    assert "y" in df.columns
    # y exists only in expB (variant 20); expA (variant 10) rows are null for y.
    assert df.loc[df["variant"] == 10, "y"].isna().all()
    assert not df.loc[df["variant"] == 20, "y"].isna().any()


def test_generation_lower_bound_filters(tmp_path):
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    out = os.path.join(root, "campaign.parquet")
    assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root,
                      generation_lower_bound=1)
    df = _read(out)
    assert (df["generation"] >= 1).all()
    assert (df["generation"] == 0).sum() == 0


def test_computed_columns_added(tmp_path):
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    out = os.path.join(root, "campaign.parquet")
    assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root,
                      computed_columns={"x_doubled": "x * 2"})
    df = _read(out)
    assert "x_doubled" in df.columns
    ok = df.dropna(subset=["x"])
    assert (ok["x_doubled"] == ok["x"] * 2).all()


def test_provenance_carried_in_report(tmp_path):
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    out = os.path.join(root, "campaign.parquet")
    store = assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root)
    assert store.provenance[10]["experiment_prefix"] == "expA"
    assert store.provenance[10]["db_id"] == 1
    assert store.provenance[20]["arm"] == "right"


def test_missing_prefix_fails_loud(tmp_path):
    """A manifest row whose store has no history parquet must raise, not silently
    produce a short store — the exact bug B4 exists to kill."""
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    manifest["rows"].append(
        {"variant": 30, "experiment_prefix": "expDOESNOTEXIST", "db_id": 3}
    )
    out = os.path.join(root, "campaign.parquet")
    with pytest.raises(CampaignAssemblyError) as ei:
        assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root)
    assert "30" in str(ei.value) or "expDOESNOTEXIST" in str(ei.value)


def test_duplicate_variant_rejected(tmp_path):
    """Two manifest rows claiming the same variant is a manifest error (ambiguous
    assignment) — fail loud."""
    root = str(tmp_path)
    manifest = _make_two_store_campaign(root)
    manifest["rows"][1]["variant"] = 10  # collide with row 0
    out = os.path.join(root, "campaign.parquet")
    with pytest.raises(CampaignAssemblyError):
        assemble_campaign(manifest, columns=["x", "y"], out_path=out, base=root)
