"""Run identity + code provenance, importable from library code.

These two helpers used to live in ``scripts/`` (``_run_provenance.py``,
``_provenance.py``), which made them unavailable to anything under
``v2ecoli/`` — ``scripts/`` is not a package. The cache in
``sim_vector_cache`` needs both to stamp what it writes, so they live here
now and the script modules re-export them for their existing callers.

Run identity follows ``docs/conventions/run-provenance.md``:
``experiment_id == run_id == the top parquet partition key``.
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import warnings
from pathlib import Path

_S3_PREFIX = "s3://"


def run_id_from_run_dir(run_dir: str) -> str:
    """Return the canonical run_id (``experiment_id``) for ``run_dir``.

    1. Authoritative: an ``experiment_id=<id>`` partition segment anywhere
       under the run dir — literally the key the emitter wrote.
    2. Fallback: the run dir's basename (runners name
       ``out/<experiment_id>/<experiment_id>/``, so the basename is the id).

    For an ``s3://`` sweep only the fallback applies: globbing the remote
    layout would cost a LIST per call, and the basename is the id by the
    same runner convention.
    """
    run_dir = run_dir.rstrip("/")
    if run_dir.startswith(_S3_PREFIX):
        return os.path.basename(run_dir)
    for cfg in glob.glob(os.path.join(run_dir, "**", "experiment_id=*"),
                         recursive=True):
        seg = os.path.basename(cfg)
        if seg.startswith("experiment_id="):
            return seg.split("=", 1)[1]
    return os.path.basename(run_dir)


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
            f"not a clean-commit claim")
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None,
        "untracked": n_untracked,
    }
