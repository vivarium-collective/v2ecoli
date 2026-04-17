"""Pickle/dill/cloudpickle allowlist.

AGENTS.md:112-114 bans pickle, dill, and cloudpickle from save-state paths.
The only exceptions are ParCa-cache and parameter-compilation artifacts.

This test scans `v2ecoli/` for any import of pickle/dill/cloudpickle and
fails if a file appears that isn't in ALLOWLIST (for individual files) or
under an allowlisted package prefix. It catches three kinds of import:

  * ``import pickle``
  * ``from pickle import ...``
  * ``__import__('pickle')``  (and same for dill, cloudpickle)

To legitimately add an import, add the path to ALLOWLIST with a one-line
justification — future reviewers can judge whether the exception holds.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.fast


REPO_ROOT = Path(__file__).resolve().parent.parent
V2ECOLI_DIR = REPO_ROOT / 'v2ecoli'

BANNED = frozenset({'pickle', 'dill', 'cloudpickle'})

# File-level allowlist (paths relative to v2ecoli/). Value explains why.
ALLOWLIST: dict[str, str] = {
    'composite.py':
        'loads ParCa sim_data_cache.dill (ParCa cache — AGENTS.md:114 exception)',
    'bridge.py':
        'loads ParCa sim_data_cache.dill',
    'library/sim_data.py':
        'loads ParCa-produced sim_data pickle; dill for compiled rate fns',
    'library/ecoli_step.py':
        'loads config_defaults.pickle (ParCa-derived defaults)',
    'library/function_registry.py':
        'base64-dill for compiled rate functions (parameter compilation)',
}

# Package-prefix allowlist (paths starting with these are ParCa internal,
# blanket-allowed per AGENTS.md:113).
ALLOWED_PREFIXES: tuple[str, ...] = (
    'processes/parca/',
)


def _banned_imports(tree: ast.AST) -> list[str]:
    """Return the sorted set of BANNED module names imported anywhere in
    the AST. Catches top-level and nested imports as well as __import__."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root in BANNED:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split('.')[0]
                if root in BANNED:
                    found.add(root)
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == '__import__' and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    root = arg.value.split('.')[0]
                    if root in BANNED:
                        found.add(root)
    return sorted(found)


def _scan() -> dict[str, list[str]]:
    """Return {relative_path: [imported_banned_modules]} for every .py
    file under v2ecoli/ that imports at least one banned module."""
    offenders: dict[str, list[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(V2ECOLI_DIR):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            full = Path(dirpath) / fn
            rel = str(full.relative_to(V2ECOLI_DIR))
            try:
                tree = ast.parse(full.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            banned = _banned_imports(tree)
            if banned:
                offenders[rel] = banned
    return offenders


def _is_allowed(rel_path: str) -> bool:
    norm = rel_path.replace(os.sep, '/')
    if norm in ALLOWLIST:
        return True
    return any(norm.startswith(prefix) for prefix in ALLOWED_PREFIXES)


# Scan once at module load; both tests query the same result.
_OFFENDERS: dict[str, list[str]] = _scan()


def test_no_new_pickle_dill_cloudpickle_imports():
    """Every file under v2ecoli/ that imports pickle/dill/cloudpickle must
    be in ALLOWLIST or under an allowlisted ParCa prefix. New offenders
    mean either (a) a new ParCa-adjacent file that should be added to
    ALLOWLIST with a justification, or (b) a save-state path regression
    that must be rewritten onto the bigraph-schema JSON API."""
    unexpected = {p: m for p, m in _OFFENDERS.items() if not _is_allowed(p)}
    assert not unexpected, (
        'New files import pickle/dill/cloudpickle without being on the '
        'allowlist. Either rewrite using v2ecoli.cache (JSON) or add the '
        'path to ALLOWLIST in this file with a one-line justification:\n'
        + '\n'.join(f'  {p}: imports {mods}' for p, mods in unexpected.items())
    )


def test_allowlist_has_no_stale_entries():
    """Every ALLOWLIST entry must still import a banned module. Stale
    entries mean someone removed the pickle usage but left the exception
    behind — clean them up so the allowlist stays informative."""
    stale = [path for path in ALLOWLIST if path not in _OFFENDERS]
    assert not stale, (
        'ALLOWLIST contains paths that no longer import any banned module. '
        'Remove them to keep the allowlist meaningful:\n'
        + '\n'.join(f'  {p}' for p in stale)
    )
