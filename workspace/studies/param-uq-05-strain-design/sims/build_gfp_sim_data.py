#!/usr/bin/env python
"""Build a GFP new-gene sim_data for param-uq-05-strain-design.

Constructs the classic reconstruction KnowledgeBaseEcoli with
``new_genes_option='gfp'``, runs ``fitSimData_1``, and:

  1. dills the fitted ``SimulationDataEcoli`` object to
     ``<outdir>/gfp_sim_data.dill`` (the driver deep-copies + mutates this
     per UQ sample), and
  2. writes a v2ecoli simulation-input bundle via ``save_sim_input`` to
     ``<outdir>/cache`` (the baseline() cache layout), as a build sanity check.

GATE: after the fit, load the object and confirm
``get_new_gene_ids_and_indices(sim_data)`` returns a NON-EMPTY GFP
cistron/monomer set. Prints the ids. Exits non-zero if empty.

Usage (from a v2ecoli worktree, with its .venv):
    python build_gfp_sim_data.py --outdir /path/out/paramuq05 --cpus 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import dill


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--debug", action="store_true",
                    help="fitSimData_1 debug mode (fit one TF; FAST but not "
                         "production-calibrated). Default off (full fit).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
    from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
    from ecoli.variants.new_gene_internal_shift import get_new_gene_ids_and_indices

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building raw KB (new_genes_option='gfp') ...",
          flush=True)
    raw = KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
        new_genes_option="gfp",
    )
    print(f"    raw KB built in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Running fitSimData_1 "
          f"(cpus={args.cpus}, debug={args.debug}) ...", flush=True)
    cache_dir_fit = os.path.join(args.outdir, "km_cache")
    os.makedirs(cache_dir_fit, exist_ok=True)
    # kwargs mirror vEcoli runscripts/parca.py run_parca (configs/default.json).
    sim_data = fitSimData_1(
        raw_data=raw,
        cpus=args.cpus,
        debug=args.debug,
        load_intermediate=None,
        save_intermediates=False,
        intermediates_directory="",
        variable_elongation_transcription=True,
        variable_elongation_translation=False,
        disable_ribosome_capacity_fitting=False,
        disable_rnapoly_capacity_fitting=False,
        cache_dir=cache_dir_fit,
        rnaseq_manifest_path=None,
        rnaseq_basal_dataset_id=None,
        basal_expression_condition="M9 Glucose minus AAs",
        rnaseq_fill_missing_genes_from_ref=True,
    )
    print(f"    fitSimData_1 done in {time.time() - t1:.1f}s", flush=True)

    # ---- GATE ----------------------------------------------------------
    cis_ids, cis_idx, mon_ids, mon_idx = get_new_gene_ids_and_indices(sim_data)
    print("\n==== GFP new-gene GATE ====", flush=True)
    print(f"  new_gene_cistron_ids : {cis_ids}", flush=True)
    print(f"  new_gene_indices     : {cis_idx}", flush=True)
    print(f"  new_monomer_ids      : {mon_ids}", flush=True)
    print(f"  new_monomer_indices  : {mon_idx}", flush=True)
    if not cis_ids or not mon_ids:
        print("GATE FAILED: empty new-gene set", flush=True)
        sys.exit(2)
    print("GATE PASSED: non-empty GFP cistron/monomer set", flush=True)

    # ---- dill the sim_data object --------------------------------------
    dill_path = os.path.join(args.outdir, "gfp_sim_data.dill")
    t2 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Dilling sim_data -> {dill_path} ...",
          flush=True)
    with open(dill_path, "wb") as f:
        dill.dump(sim_data, f, protocol=dill.HIGHEST_PROTOCOL)
    print(f"    dilled in {time.time() - t2:.1f}s "
          f"({os.path.getsize(dill_path) / 1e6:.1f} MB)", flush=True)

    # ---- v2ecoli bundle (sanity check that baseline can consume it) ----
    cache_dir = os.path.join(args.outdir, "cache")
    t3 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] save_sim_input -> {cache_dir} ...",
          flush=True)
    from v2ecoli.core import save_sim_input
    save_sim_input(sim_data, cache_dir)
    print(f"    bundle written in {time.time() - t3:.1f}s", flush=True)
    for fn in sorted(os.listdir(cache_dir)):
        p = os.path.join(cache_dir, fn)
        if os.path.isfile(p):
            print(f"      {fn}: {os.path.getsize(p) / 1e6:.2f} MB", flush=True)

    print(f"\nTotal: {time.time() - t0:.1f}s", flush=True)
    print("BUILD OK", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
