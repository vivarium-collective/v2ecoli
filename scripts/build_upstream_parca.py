"""Build a pristine upstream ``CovertLab/vEcoli`` ParCa ``simData.cPickle``.

The ``--composite vecoli`` engine (the external upstream wrapper in
:mod:`v2ecoli.library.vecoli_pbg_upstream`) needs an UPSTREAM-MASTER-built
``sim_data`` — the v2ecoli-built ``out/kb`` pickle predates upstream master's
TCS ``modified_molecules`` and won't unpickle cleanly under master. This script
builds one from the pristine upstream checkout (``$V2E_VECOLI_DIR``, default
``/app/vEcoli`` in the GovCloud image) by calling upstream's
``runscripts.parca.run_parca`` directly.

It reuses the wrapper's :func:`~v2ecoli.library.vecoli_pbg_upstream._ensure_upstream`
import shim — pin the compiled ``wholecell`` tree, put the source-only upstream
checkout on ``sys.path``, make vivarium's registry idempotent — then purges any
cached ``runscripts``/``reconstruction``/``validation`` modules so they resolve
from the upstream checkout (NOT a shadow install).

Output: ``<outdir>/kb/simData.cPickle`` (upstream's ``run_parca`` layout). Pass
``--copy-to <dir>`` to also drop a flat copy at ``<dir>/simData.cPickle`` — the
layout the harness ``--cache-dir`` expects (the vivarium-process vEcoli loader
reads ``os.path.join(cache_dir, "simData.cPickle")``).

Usage (run on Linux / inside the image; heavy — minutes):

    V2E_VECOLI_DIR=/app/vEcoli python scripts/build_upstream_parca.py \
        --outdir /app/vEcoli/out --cpus 8 --copy-to /app/out/cache

NOTE: on Linux (GovCloud) multiprocessing defaults to ``fork``, which inherits
the already-pinned compiled ``wholecell`` into ParCa's worker pool, so the macOS
``spawn`` re-import failure does NOT apply there. We still force ``fork``
defensively. Do NOT run this on macOS (spawn workers re-import and break the
wholecell pin; macOS is dev-only here).

This calls ``run_parca(cfg)`` DIRECTLY rather than ``runscripts.parca.main()``:
``main()`` short-circuits (skip + copy) whenever the config carries a top-level
``sim_data_path``, which is not what we want here — we want a fresh fit.

Optional ``--config PATH`` (item 87): without it, this builds EXACTLY the same
pristine baseline sim_data as before -- ``configs/default.json``'s own
``parca_options`` unchanged. When given, that config's OWN ``parca_options``
(e.g. ``new_genes``) are merged on top of ``default.json``'s, the same
override-wins-on-conflict shape the config itself already carries at every
other layer (``EcoliSim``'s own inheritance loader). This does NOT stage any
new-gene DATA files -- ``new_genes`` names a subdirectory under
``reconstruction/ecoli/flat/new_gene_data/`` that must already exist in the
checkout; a config requesting a subdirectory that isn't there fails loudly
inside ``KnowledgeBaseEcoli`` (an assertion), not silently.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _resolve_parca_options(default_cfg: dict, config_path: str | None) -> dict:
    """Merge an optional override config's ``parca_options`` onto the default's.

    ``config_path`` is None for every existing caller -- returns
    ``default_cfg["parca_options"]`` completely unchanged (same object shape
    and values as before this function existed). When given, the override
    config's own ``parca_options`` keys win on conflict (the only keys a
    caller would set are ones it deliberately wants to differ from the
    baseline, e.g. ``new_genes``); keys the override config doesn't declare
    keep the baseline's value. Pure function, no filesystem/import side
    effects beyond reading ``config_path`` -- independently testable without
    the upstream checkout.
    """
    parca_options = dict(default_cfg["parca_options"])
    if config_path is None:
        return parca_options
    with open(config_path) as fh:
        override_cfg = json.load(fh)
    parca_options.update(override_cfg.get("parca_options") or {})
    return parca_options


def build_upstream_parca(
    outdir: str, cpus: int, copy_to: str | None = None, config: str | None = None
) -> str:
    """Build upstream sim_data into ``<outdir>/kb/simData.cPickle``; return its path."""
    # Defensive on Linux: fork inherits the pinned compiled wholecell into the
    # ParCa worker pool (spawn would re-import the source-only checkout copy and
    # raise "Failed to import Cython module"). No-op where fork is already default.
    try:
        multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    from v2ecoli.library.vecoli_pbg_upstream import _ensure_upstream

    up = _ensure_upstream()
    vecoli_dir = up["vecoli_dir"]

    # Purge any cached runscripts/reconstruction/validation modules (possibly
    # from a shadow install) so run_parca's imports resolve from the upstream
    # checkout that _ensure_upstream put first on sys.path.
    for _m in [
        m for m in list(sys.modules)
        if m.split(".")[0] in ("runscripts", "reconstruction", "validation")
    ]:
        del sys.modules[_m]

    # Upstream's run_parca + configs.default both resolve relative to the
    # checkout (configs/ is a top-level package there; KnowledgeBaseEcoli reads
    # flat data under reconstruction/). chdir into it like the wrapper does.
    prev_cwd = os.getcwd()
    os.chdir(vecoli_dir)
    try:
        from runscripts.parca import run_parca

        outdir = os.path.abspath(outdir)
        default_cfg = json.load(open(os.path.join(vecoli_dir, "configs", "default.json")))
        parca_options = _resolve_parca_options(default_cfg, config)
        parca_options["outdir"] = outdir
        parca_options["cpus"] = int(cpus)
        parca_options["cache_dir"] = os.path.join(outdir, "cache")
        os.makedirs(parca_options["cache_dir"], exist_ok=True)

        print(f"[build_upstream_parca] outdir={outdir} cpus={cpus} "
              f"vecoli_dir={vecoli_dir} config={config or 'default'} "
              f"new_genes={parca_options.get('new_genes', 'off')}", flush=True)
        sim_data_hash = run_parca(parca_options)
        print(f"[build_upstream_parca] run_parca OK sha256={sim_data_hash}", flush=True)
    finally:
        os.chdir(prev_cwd)

    sim_data_path = os.path.join(outdir, "kb", "simData.cPickle")
    if not os.path.exists(sim_data_path):
        raise SystemExit(f"ParCa finished but {sim_data_path} is missing")
    print(f"[build_upstream_parca] sim_data -> {sim_data_path}", flush=True)

    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        dest = os.path.join(copy_to, "simData.cPickle")
        shutil.copy2(sim_data_path, dest)
        print(f"[build_upstream_parca] flat copy -> {dest} "
              f"(harness --cache-dir layout)", flush=True)
    return sim_data_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    default_vecoli = os.environ.get("V2E_VECOLI_DIR", "/app/vEcoli")
    p.add_argument("--outdir", default=os.path.join(default_vecoli, "out"),
                   help="ParCa output root; sim_data lands at <outdir>/kb/"
                        "simData.cPickle (default: $V2E_VECOLI_DIR/out).")
    p.add_argument("--cpus", type=int, default=int(os.environ.get("PARCA_CPUS", "8")),
                   help="ParCa fit parallelism (default: $PARCA_CPUS or 8).")
    p.add_argument("--copy-to", default=None,
                   help="Also place a flat simData.cPickle here (the harness "
                        "--cache-dir layout the entrypoint stages to S3).")
    p.add_argument("--config", default=None,
                   help="Optional full config JSON whose parca_options (e.g. "
                        "new_genes) override configs/default.json's -- omit "
                        "for the pristine baseline build (unchanged default "
                        "behavior). The named new_genes subdirectory must "
                        "already exist under reconstruction/ecoli/flat/"
                        "new_gene_data/ in the checkout; this flag does not "
                        "stage any data files itself.")
    args = p.parse_args(argv)
    build_upstream_parca(args.outdir, args.cpus, args.copy_to, args.config)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
