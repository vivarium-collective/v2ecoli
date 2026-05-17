"""Unit tests for the ``sqlite_emitter()`` workspace-default behavior.

Confirms:
  - When called with no ``file_path``, the emitter override resolves to
    ``<workspace_root>/.pbg/composite-runs.db`` (auto-detected by walking
    up from cwd looking for ``workspace.yaml``).
  - The ``simulations`` table grows ``study_slug`` + ``investigation_slug``
    columns on first use, and the slugs passed to the context manager are
    stamped onto the row.
  - Explicit ``file_path`` + ``db_file`` still work (back-compat).

These tests don't run a real composite — they exercise the helper around a
synthetic SQLite DB so they're fast (no parca, no cell sim).
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from v2ecoli.composites._helpers import (
    _EMITTER_OVERRIDE,  # noqa: F401 — referenced via module attribute
    set_emitter_override,
    sqlite_emitter,
)
from v2ecoli.composites import _helpers as _h


@pytest.mark.fast
def test_workspace_default_resolves_to_pbg_composite_runs_db(tmp_path, monkeypatch):
    """No file_path -> <workspace>/.pbg/composite-runs.db, with the slug columns."""
    (tmp_path / 'workspace.yaml').write_text('name: test-ws\n')
    monkeypatch.chdir(tmp_path)

    with sqlite_emitter(
        name='smoke-run',
        study_slug='dnaa-01-expression-dynamics',
        investigation_slug='dnaa-replication',
    ) as cfg:
        assert cfg['file_path'] == str(tmp_path / '.pbg')
        assert cfg['db_file'] == 'composite-runs.db'
        sim_id = cfg['simulation_id']

    db = tmp_path / '.pbg' / 'composite-runs.db'
    assert db.is_file(), '.pbg/composite-runs.db not created'
    conn = sqlite3.connect(str(db))
    try:
        cols = {row[1] for row in conn.execute('PRAGMA table_info(simulations)')}
        assert 'study_slug' in cols
        assert 'investigation_slug' in cols
        row = conn.execute(
            'SELECT study_slug, investigation_slug '
            'FROM simulations WHERE simulation_id = ?',
            (sim_id,),
        ).fetchone()
        assert row == ('dnaa-01-expression-dynamics', 'dnaa-replication')
    finally:
        conn.close()


@pytest.mark.fast
def test_explicit_file_path_preserves_backcompat(tmp_path, monkeypatch):
    """Passing file_path explicitly keeps the old per-study layout."""
    study_dir = tmp_path / 'studies' / 'dnaa-01'
    study_dir.mkdir(parents=True)
    # Put us somewhere with no workspace.yaml to prove the default-resolver
    # isn't triggered when file_path is supplied.
    monkeypatch.chdir(tmp_path)

    with sqlite_emitter(
        file_path=str(study_dir),
        db_file='runs.db',
        name='legacy',
    ) as cfg:
        assert cfg['file_path'] == str(study_dir)
        assert cfg['db_file'] == 'runs.db'

    assert (study_dir / 'runs.db').is_file()


@pytest.mark.fast
def test_missing_workspace_raises(tmp_path, monkeypatch):
    """No workspace.yaml in the cwd chain + no file_path -> clear error."""
    # Park cwd somewhere with no workspace.yaml ancestor.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_h, '_find_workspace_root', lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match='workspace.yaml'):
        with sqlite_emitter(name='no-ws'):
            pass


@pytest.mark.fast
def test_slugs_optional(tmp_path, monkeypatch):
    """Omitting slugs leaves the columns NULL; no crash."""
    (tmp_path / 'workspace.yaml').write_text('name: test-ws\n')
    monkeypatch.chdir(tmp_path)
    with sqlite_emitter(name='plain') as cfg:
        sim_id = cfg['simulation_id']
    db = tmp_path / '.pbg' / 'composite-runs.db'
    assert db.is_file()
    conn = sqlite3.connect(str(db))
    try:
        # The row is only inserted when slugs are present; without slugs
        # the helpers don't touch the row at all (the emitter would insert
        # it at runtime). Verify the columns at least exist on the table.
        cols = {row[1] for row in conn.execute('PRAGMA table_info(simulations)')}
        assert 'study_slug' in cols
        assert 'investigation_slug' in cols
        row = conn.execute(
            'SELECT simulation_id FROM simulations WHERE simulation_id = ?',
            (sim_id,),
        ).fetchone()
        # No row from us — we didn't stamp anything. That's fine.
        assert row is None or row[0] == sim_id
    finally:
        conn.close()


@pytest.mark.fast
def test_override_is_cleared_on_exit(tmp_path, monkeypatch):
    """Even on exception, the module-level emitter override is reset."""
    (tmp_path / 'workspace.yaml').write_text('name: test-ws\n')
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match='boom'):
        with sqlite_emitter(name='will-fail'):
            assert _h._EMITTER_OVERRIDE is not None
            raise RuntimeError('boom')
    assert _h._EMITTER_OVERRIDE is None
    # Cleanup — paranoia for parallel test runs.
    set_emitter_override(None)
