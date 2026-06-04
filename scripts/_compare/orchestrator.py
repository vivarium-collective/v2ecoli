"""Cache-aware subprocess wrappers that run each engine's ParCa and sim."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts._compare.cache import is_stale, mark_done

VECOLI_REPO = "/Users/eranagmon/code/vEcoli"
VECOLI_PYTHON = f"{VECOLI_REPO}/.venv/bin/python"
V2_PYTHON = ".venv/bin/python"


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd)
    if getattr(proc, "returncode", 0) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}")


def run_v2_parca(*, out_dir: Path, cache_dir: Path, mode: str) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run(["v2ecoli-parca", "--mode", mode, "-o", str(out_dir),
          "--cache-dir", str(cache_dir)])
    mark_done(out_dir)
    return out_dir


def run_vecoli_parca(*, config_path: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([VECOLI_PYTHON, "runscripts/parca.py",
          "--config", config_path,
          "--outdir", str(out_dir),
          "--save-intermediates",
          "--intermediates-directory", str(out_dir)],
         cwd=VECOLI_REPO)
    mark_done(out_dir)
    return out_dir


def run_vecoli_sim(*, config_path: str, sim_data_path: str, out_dir: Path,
                   generations: int = 2) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([VECOLI_PYTHON, "ecoli/experiments/ecoli_master_sim.py",
          "--config", config_path,
          "--generations", str(generations),
          "--emitter", "parquet",
          "--emitter_arg", f"out_dir={out_dir}",
          "--sim_data_path", sim_data_path],
         cwd=VECOLI_REPO)
    mark_done(out_dir)
    return out_dir


def run_v2_sim(*, config_path: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([V2_PYTHON, "-m", "v2ecoli.workflow.run",
          "--config", config_path, "--out", str(out_dir)])
    mark_done(out_dir)
    return out_dir
