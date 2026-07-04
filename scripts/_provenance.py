"""Stamp a reproducibility-provenance block onto a baked golden fixture.

Golden model fixtures (``tests/fixtures/<card>/model_*.json``) are organized
per-cell aggregates from one particular sweep — not general reference data. To
trace a fixture back to the exact state that produced it, each bake stamps:

  * code identity — the commit, PLUS a ``dirty`` flag and a sha256 of ``git diff
    HEAD`` when the tree is dirty. A bare commit is only honest from a clean tree;
    fixtures are usually baked mid-iteration, so the commit alone would claim a
    clean provenance that is a lie. ``diff_sha256`` makes a dirty bake at least
    *identifiable* (re-derive ``git diff HEAD | sha256sum`` and compare), which
    ``git describe --dirty`` cannot. Untracked files are flagged by count (they
    don't appear in ``git diff``).
  * the config, the source-sweep identity, the bake command, and the date.

``reconstructed_stamp`` writes an honest "we don't actually know" block for
fixtures whose source state was never recorded (backfill); re-bake for an
authoritative stamp.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
import warnings
from pathlib import Path


def _git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def code_provenance(repo: Path) -> dict:
    """Commit + dirty/diff-sha + untracked count for the working tree at ``repo``.

    Warns when dirty/untracked so a mid-dev bake is loud about its provenance
    being identify-only rather than a clean commit."""
    commit = _git(["rev-parse", "HEAD"], repo)
    diff = _git(["diff", "HEAD"], repo) or ""          # tracked staged+unstaged vs HEAD
    untracked = _git(["ls-files", "--others", "--exclude-standard"], repo) or ""
    dirty = bool(diff)
    n_untracked = len([u for u in untracked.splitlines() if u])
    if dirty or n_untracked:
        warnings.warn(
            f"baking from a DIRTY tree (commit {commit[:12] if commit else '?'}, "
            f"dirty={dirty}, untracked={n_untracked}) — provenance is identify-only, "
            "not a clean commit. Commit first for a clean-provenance fixture.")
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None,
        "untracked": n_untracked,
    }


def provenance_stamp(repo: Path, *, config: str, sweep: dict,
                     bake_script: str) -> dict:
    """Authoritative provenance for a freshly-baked fixture."""
    return {
        "code": code_provenance(repo),
        "config": config,
        "sweep": sweep,
        "bake_script": bake_script,
        "baked": time.strftime("%Y-%m-%d"),
        "reconstructed": False,
    }


def reconstructed_stamp(*, config: str, sweep: dict, note: str) -> dict:
    """Honest backfill for a fixture whose source state was never recorded.

    ``code`` is null (the producing commit/diff is unknown — stamping the current
    HEAD would falsely claim it baked these values)."""
    return {
        "code": {"commit": None, "dirty": None, "diff_sha256": None, "untracked": None},
        "config": config,
        "sweep": sweep,
        "bake_script": None,
        "baked": "unknown",
        "reconstructed": True,
        "note": note,
    }
