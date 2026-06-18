"""Cache-aware subprocess wrappers that run each engine's ParCa and sim.

Each engine runs against its OWN venv. The vEcoli side (ParCa scripts and the
Nextflow workflow, whose tasks spawn a bare ``python``) is run with vEcoli's
``.venv/bin`` prepended to ``PATH`` so those child processes resolve vEcoli's
interpreter/packages — NOT whatever venv launched the harness. The v2ecoli side
invokes its console scripts by absolute path inside its own venv. This keeps the
two engines' (incompatible) ``wholecell``/dependency trees from leaking across.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts._compare.cache import is_stale, mark_done

VECOLI_REPO = "/Users/eranagmon/code/vEcoli"
VECOLI_PYTHON = f"{VECOLI_REPO}/.venv/bin/python"
V2_PYTHON = ".venv/bin/python"
# v2ecoli console scripts live next to the v2 interpreter; resolve by absolute
# path so the harness needn't pollute PATH (which would leak into vEcoli tasks).
V2_BIN = str(Path(V2_PYTHON).parent)


def _run(cmd, cwd=None, env=None):
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if getattr(proc, "returncode", 0) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}")


def _vecoli_env(vecoli_repo: str) -> dict:
    """Environment with vEcoli's venv first on PATH, so Nextflow's spawned
    ``python`` tasks use vEcoli's interpreter rather than the harness launcher's."""
    path = os.environ.get("PATH", "")
    return {**os.environ, "PATH": f"{vecoli_repo}/.venv/bin:{path}"}


def run_v2_parca(*, out_dir: Path, cache_dir: Path, mode: str,
                 token: str | None = None) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    _run([f"{V2_BIN}/v2ecoli-parca", "--mode", mode, "-o", str(out_dir),
          "--cache-dir", str(cache_dir)])
    mark_done(out_dir, token or "ok")
    return out_dir


def run_vecoli_parca(*, config_path: str, out_dir: Path,
                     token: str | None = None,
                     vecoli_repo: str = VECOLI_REPO) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    vecoli_python = f"{vecoli_repo}/.venv/bin/python"
    _run([vecoli_python, "runscripts/parca.py",
          "--config", config_path,
          "--outdir", str(out_dir),
          "--save-intermediates",
          "--intermediates-directory", str(out_dir)],
         cwd=vecoli_repo, env=_vecoli_env(vecoli_repo))
    mark_done(out_dir, token or "ok")
    return out_dir


def run_vecoli_sim(*, config_path: str, out_dir: Path,
                   token: str | None = None,
                   vecoli_repo: str = VECOLI_REPO) -> Path:
    """Run vEcoli's Nextflow workflow for the 2-gen lineage.

    The config is expected to set ``sim_data_path`` (so the workflow copies
    that kb and skips re-running ParCa) and ``out_dir``/emitter. Reads all
    run parameters from the config JSON, mirroring run_v2_sim.

    Runs with vEcoli's venv on PATH so the Nextflow tasks' bare ``python``
    resolves vEcoli's interpreter (its ``wholecell`` differs from v2ecoli's).
    """
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    vecoli_python = f"{vecoli_repo}/.venv/bin/python"
    _run([vecoli_python, "-m", "runscripts.workflow", "--config", config_path],
         cwd=vecoli_repo, env=_vecoli_env(vecoli_repo))
    mark_done(out_dir, token or "ok")
    return out_dir


def run_v2_sim(*, config_path: str, out_dir: Path,
               token: str | None = None) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    _run([V2_PYTHON, "-m", "v2ecoli.workflow.run",
          "--config", config_path, "--out", str(out_dir)])
    mark_done(out_dir, token or "ok")
    return out_dir
