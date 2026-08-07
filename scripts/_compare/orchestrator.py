"""Cache-aware subprocess wrappers that run each engine's ParCa and sim.

Each engine runs against its OWN venv. The vEcoli side (ParCa scripts and the
Nextflow workflow, whose tasks spawn a bare ``python``) is run with vEcoli's
``.venv/bin`` prepended to ``PATH`` so those child processes resolve vEcoli's
interpreter/packages — NOT whatever venv launched the harness. The v2ecoli side
invokes its console scripts by absolute path inside its own venv. This keeps the
two engines' (incompatible) ``wholecell``/dependency trees from leaking across.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts._compare.cache import is_stale, mark_done

V2_PYTHON = ".venv/bin/python"
# v2ecoli console scripts live next to the v2 interpreter; resolve by absolute
# path so the harness needn't pollute PATH (which would leak into vEcoli tasks).
V2_BIN = str(Path(V2_PYTHON).parent)


def _run(cmd, cwd=None, env=None, retries=0):
    """Run ``cmd``; on non-zero exit retry up to ``retries`` more times.

    Each attempt is a fresh invocation. The vEcoli workflow stamps every run
    with a new timestamped nextflow_temp / out dir, so a retry re-runs cleanly
    rather than resuming partial state — this recovers the transient Nextflow
    launch/JVM/resource failures that otherwise drop a single seed from a
    multi-seed batch.
    """
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, cwd=cwd, env=env)
        rc = getattr(proc, "returncode", 0)
        if rc == 0:
            return
        last = rc
        if attempt < retries:
            print(f"  [orchestrator] command failed (rc={rc}); "
                  f"retry {attempt + 1}/{retries}: {cmd[0]} ...")
    raise RuntimeError(
        f"command failed (rc={last}) after {retries + 1} attempt(s): {cmd}")


def run_v2_parca(*, out_dir: Path, cache_dir: Path, mode: str,
                 token: str | None = None) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    _run([f"{V2_BIN}/v2ecoli-parca", "--mode", mode, "-o", str(out_dir),
          "--cache-dir", str(cache_dir)])
    mark_done(out_dir, token or "ok")
    return out_dir


def run_vecoli_parca(*, reference, config_path: str, out_dir: Path,
                     token: str | None = None) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    _run(reference.parca_cmd(config_path, str(out_dir), str(out_dir)),
         cwd=reference.repo, env=reference.env())
    mark_done(out_dir, token or "ok")
    return out_dir


def run_vecoli_sim(*, reference, config_path: str, out_dir: Path,
                   token: str | None = None,
                   render_only: bool = False) -> Path:
    """Run vEcoli's Nextflow workflow for the 2-gen lineage.

    The config is expected to set ``sim_data_path`` (so the workflow copies
    that kb and skips re-running ParCa) and ``out_dir``/emitter. Reads all
    run parameters from the config JSON, mirroring run_v2_sim.

    Runs with the reference engine's venv on PATH so the Nextflow tasks' bare
    ``python`` resolves the reference's interpreter (its ``wholecell`` differs
    from v2ecoli's).

    ``render_only`` re-uses whatever is already on disk WITHOUT running (and
    without the staleness check) — the report is rebuilt from the existing
    runs. A current run is skipped automatically via ``is_stale`` regardless.
    """
    out_dir = Path(out_dir)
    if render_only or not is_stale(out_dir, token):
        return out_dir
    # Nextflow occasionally fails to launch (JVM/resource hiccup); retry so a
    # transient failure on one seed doesn't drop it from a multi-seed batch.
    _run(reference.sim_cmd(config_path), cwd=reference.repo,
         env=reference.env(), retries=2)
    mark_done(out_dir, token or "ok")
    return out_dir


def run_v2_sim(*, config_path: str, out_dir: Path,
               token: str | None = None, render_only: bool = False) -> Path:
    out_dir = Path(out_dir)
    if render_only or not is_stale(out_dir, token):
        return out_dir
    _run([V2_PYTHON, "-m", "v2ecoli.workflow.run",
          "--config", config_path, "--out", str(out_dir)], retries=1)
    mark_done(out_dir, token or "ok")
    return out_dir
