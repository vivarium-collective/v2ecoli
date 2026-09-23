#!/usr/bin/env python
"""Build a named ParCa cache with a translation-level knockout BAKED IN.

Why a cache rather than the `knockouts=` argument
-------------------------------------------------
`baseline(knockouts=[...])` resolves the knockout at build time into a
`config_overrides` entry (`ecoli_baseline.py:1414-1415`). ⚠ `division.py:373-376`
rebuilds each daughter with only `core`, `seed`, `cache_dir`, `emitter` and
`injected_processes` — `config_overrides` is NOT among them, so the knockout
applies to generation 1 and every daughter silently reverts to wild type.
Measured for trpR: mother `translation_efficiencies[3834] = 0.0`, both daughters
`2.4992e-04`. (v2ecoli#505 fixes this; it is unmerged as of 2026-08-21.)

A cache-resident perturbation has no such problem: `cache_dir` IS threaded to
the daughter rebuild, and `extend_multigen_from_dill.py:87-94` reads the process
configs from the cache once and applies them to every generation. So baking the
knockout into a cache makes it survive division today, independently of #505.

The shape mirrors `scripts/build_condition_cache.py` (hydrate -> patch ->
save_sim_input -> manifest) with a knockout as the patch. It deliberately does
NOT reuse `v2ecoli.perturbations.build_new_gene_cache` (v2ecoli#563): that
driver calls `set_new_gene_expression` and fails fast on a sim_data carrying no
new-gene cistrons. trpR is a native gene.

⚠ The knockout is applied to `translation_efficiencies_by_monomer` BEFORE the
bundle is written, so it is present in the array `get_polypeptide_initiation_config`
normalises. Zero survives L1 normalisation (0/S == 0), so a KO is exact — unlike
a strength change, where only ratios survive the cache.

Run from the workspace root:
    python workspace/studies/genotype-06-trpr-regulon/sims/build_ko_cache.py \
        --genes EG11029 --cache out/genotype-06/cache-trpR
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from v2ecoli.core import save_sim_input
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)

DEFAULT_FIXTURE = "models/parca/parca_state.pkl.gz"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def knock_out(sim_data, gene_ids: list[str]) -> dict:
    """Zero each gene's monomer translation efficiency, in place.

    Resolves gene id -> monomer via the cistron/monomer tables rather than by
    string-matching a monomer id: a gene's monomer id is not derivable from its
    frame code, and guessing it is how a knockout silently hits nothing.
    Raises if any gene fails to resolve — a KO that quietly zeroed zero
    monomers would read as a successful build.
    """
    tl = sim_data.process.translation
    monomer_ids = np.asarray(tl.monomer_data["id"]).astype(str)
    cistron_of_monomer = np.asarray(tl.monomer_data["cistron_id"]).astype(str)
    tx = sim_data.process.transcription
    cistron_ids = np.asarray(tx.cistron_data["id"]).astype(str)
    gene_of_cistron = np.asarray(tx.cistron_data["gene_id"]).astype(str)

    applied, missing = {}, []
    for gid in gene_ids:
        cis = cistron_ids[gene_of_cistron == gid]
        ix = np.where(np.isin(cistron_of_monomer, cis))[0]
        if ix.size == 0:
            missing.append(gid)
            continue
        before = [float(tl.translation_efficiencies_by_monomer[i]) for i in ix]
        tl.translation_efficiencies_by_monomer[ix] = 0.0
        applied[gid] = {"monomer_indices": [int(i) for i in ix],
                        "monomer_ids": [monomer_ids[i] for i in ix],
                        "efficiency_before": before,
                        "efficiency_after": 0.0}
    if missing:
        raise SystemExit(
            f"no monomer resolved for {missing} — nothing was knocked out. "
            "Check the gene ids are EcoCyc frame codes for PROTEIN-CODING genes.")
    return applied


def build(gene_ids: list[str], fixture: str, cache_dir: str,
          seed: int = 0, condition: str | None = None,
          fixed_media: str | None = None) -> dict:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading ParCa state {fixture} ...", flush=True)
    sim_data = hydrate_sim_data_from_state(load_parca_state(fixture))

    print(f"[{time.strftime('%H:%M:%S')}] Knocking out {gene_ids} ...", flush=True)
    applied = knock_out(sim_data, gene_ids)
    for gid, rec in applied.items():
        print(f"    {gid}: {rec['monomer_ids']} "
              f"{rec['efficiency_before']} -> 0.0", flush=True)

    # ORDER IS LOAD-BEARING: the zeros must already be in the array when the
    # bundle is extracted, or this is a wild-type cache that looks perturbed.
    print(f"[{time.strftime('%H:%M:%S')}] Writing bundle -> {cache_dir} ...", flush=True)
    save_sim_input(sim_data, cache_dir, seed=seed,
                   condition=condition, fixed_media=fixed_media)

    manifest = {
        "created_at": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "base_state": fixture,
        "seed": seed,
        "condition": condition,
        "fixed_media": fixed_media,
        "knockouts": gene_ids,
        # As ASSIGNED. Zero is exact under the cache's L1 normalisation, so
        # unlike a strength change this value is also what the run consumes.
        "applied": applied,
    }
    path = os.path.join(cache_dir, "knockouts.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    manifest -> {path}\nTotal: {time.time()-t0:.1f}s", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Optional so the MATCHED WT reference is built by this same code path:
    # same fixture, same condition, same builder, differing only in the
    # perturbation. A WT cache built by a different script would confound any
    # arm difference with a cache-construction difference.
    ap.add_argument("--genes", default="",
                    help="comma-separated EcoCyc gene ids, e.g. EG11029,EG11005. "
                         "Empty = unperturbed reference cache.")
    ap.add_argument("--cache", dest="cache_dir", required=True)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default=None,
                    help="ParCa nutrient condition for the initial state / "
                         "doubling time (e.g. with_aa; default basal)")
    ap.add_argument("--fixed-media", default=None,
                    help="media id pinned for the run (e.g. minimal_plus_amino_acids)")
    args = ap.parse_args()
    build([g.strip() for g in args.genes.split(",") if g.strip()],
          args.fixture, os.path.abspath(args.cache_dir), seed=args.seed,
          condition=args.condition, fixed_media=args.fixed_media)


if __name__ == "__main__":
    main()
