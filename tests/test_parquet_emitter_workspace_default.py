"""Unit tests for the ``parquet_emitter()`` workspace-default behavior.

Parallel to ``test_sqlite_emitter_workspace_default.py``. Confirms:

  - When called with no ``out_dir``, the override resolves to
    ``<workspace_root>/.pbg/parquet-runs/`` (auto-detected by walking up
    from cwd looking for ``workspace.yaml``).
  - Study + investigation slugs land in ``metadata`` (so hive readers can
    group runs by study without needing a sqlite simulations table).
  - The module-level override is cleared on exit (including on exception).
  - Explicit ``out_dir`` still works (back-compat).
"""

from __future__ import annotations

import pytest

from v2ecoli.composites import _helpers as _h
from v2ecoli.composites._helpers import (
    parquet_emitter,
    set_parquet_emitter_override,
)


@pytest.mark.fast
def test_workspace_default_resolves_to_pbg_parquet_runs(tmp_path, monkeypatch):
    """No out_dir -> <workspace>/.pbg/parquet-runs/, slugs land in metadata."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    with parquet_emitter(
        experiment_id="smoke-run",
        study_slug="dnaa-01-expression-dynamics",
        investigation_slug="dnaa-replication",
    ) as cfg:
        assert cfg["out_dir"] == str(tmp_path / ".pbg" / "parquet-runs")
        assert cfg["metadata"]["study_slug"] == "dnaa-01-expression-dynamics"
        assert cfg["metadata"]["investigation_slug"] == "dnaa-replication"
        # vEcoli-shaped partition layout
        assert cfg["partitioning_keys"] == [
            "experiment_id", "variant", "lineage_seed", "generation", "agent_id",
        ]


@pytest.mark.fast
def test_explicit_out_dir_preserves_backcompat(tmp_path, monkeypatch):
    """Passing out_dir explicitly skips the workspace lookup."""
    target = tmp_path / "studies" / "dnaa-01"
    target.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with parquet_emitter(out_dir=str(target), experiment_id="legacy") as cfg:
        assert cfg["out_dir"] == str(target)


@pytest.mark.fast
def test_missing_workspace_raises(tmp_path, monkeypatch):
    """No workspace.yaml in cwd chain + no out_dir -> clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_h, "_find_workspace_root", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="workspace.yaml"):
        with parquet_emitter(experiment_id="no-ws"):
            pass


@pytest.mark.fast
def test_override_is_cleared_on_exit(tmp_path, monkeypatch):
    """Even on exception, the module-level parquet override resets."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        with parquet_emitter(experiment_id="will-fail"):
            assert _h._PARQUET_EMITTER_OVERRIDE is not None
            raise RuntimeError("boom")
    assert _h._PARQUET_EMITTER_OVERRIDE is None
    set_parquet_emitter_override(None)


@pytest.mark.fast
def test_extra_metadata_merges_through(tmp_path, monkeypatch):
    """Caller-supplied extra_metadata lands alongside the preset's keys."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    with parquet_emitter(
        experiment_id="enrich",
        extra_metadata={"description": "from a tiny test", "seed_strategy": "fixed"},
    ) as cfg:
        meta = cfg["metadata"]
        assert meta["description"] == "from a tiny test"
        assert meta["seed_strategy"] == "fixed"
        # Preset's required partition keys still present
        assert meta["experiment_id"] == "enrich"
        assert "variant" in meta
        assert "generation" in meta
