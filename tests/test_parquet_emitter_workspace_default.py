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
    ) as emit:
        cfg = emit.cfg
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

    with parquet_emitter(out_dir=str(target), experiment_id="legacy") as emit:
        assert emit.cfg["out_dir"] == str(target)


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
def test_bind_auto_flushes_on_clean_exit(tmp_path, monkeypatch):
    """Auto-flush fires on clean exit when composite is bound. Closes
    friction-#3 (parquet context manager couldn't enforce lifecycle).
    Uses a stub composite + monkeypatched flush_parquet so we don't need
    to spin up a real ParquetEmitter for the lifecycle check."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(_h, "flush_parquet",
                        lambda comp, *, success=True: (calls.append(("clean", comp, success)) or 1))

    stub = object()
    with parquet_emitter(experiment_id="bind-clean") as emit:
        emit.bind(stub)
    assert calls == [("clean", stub, True)]


@pytest.mark.fast
def test_bind_auto_flushes_with_failure_on_exception(tmp_path, monkeypatch):
    """Exception inside the with-block → auto-flush with success=False so
    the sentinel honestly reflects the failed run."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(_h, "flush_parquet",
                        lambda comp, *, success=True: (calls.append(("exc", comp, success)) or 1))

    stub = object()
    with pytest.raises(RuntimeError, match="kapow"):
        with parquet_emitter(experiment_id="bind-exc") as emit:
            emit.bind(stub)
            raise RuntimeError("kapow")
    assert calls == [("exc", stub, False)]


@pytest.mark.fast
def test_no_bind_no_auto_flush(tmp_path, monkeypatch):
    """Legacy behaviour preserved: without .bind(), no auto-flush fires.
    Callers that already invoke flush_parquet() explicitly keep working."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(_h, "flush_parquet",
                        lambda comp, *, success=True: (calls.append("never") or 1))

    with parquet_emitter(experiment_id="no-bind") as _emit:
        pass
    assert calls == []


@pytest.mark.fast
def test_explicit_flush_skips_auto_flush(tmp_path, monkeypatch):
    """Calling .flush() inside the with-block sets _flushed; exit auto-flush
    skips. Prevents a double-close."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(_h, "flush_parquet",
                        lambda comp, *, success=True: (calls.append(success) or 1))

    stub = object()
    with parquet_emitter(experiment_id="explicit-flush") as emit:
        emit.bind(stub)
        emit.flush(success=True)
    assert calls == [True]  # only the explicit call, no second from exit


@pytest.mark.fast
def test_extra_metadata_merges_through(tmp_path, monkeypatch):
    """Caller-supplied extra_metadata lands alongside the preset's keys."""
    (tmp_path / "workspace.yaml").write_text("name: test-ws\n")
    monkeypatch.chdir(tmp_path)

    with parquet_emitter(
        experiment_id="enrich",
        extra_metadata={"description": "from a tiny test", "seed_strategy": "fixed"},
    ) as emit:
        meta = emit.cfg["metadata"]
        assert meta["description"] == "from a tiny test"
        assert meta["seed_strategy"] == "fixed"
        # Preset's required partition keys still present
        assert meta["experiment_id"] == "enrich"
        assert "variant" in meta
        assert "generation" in meta
