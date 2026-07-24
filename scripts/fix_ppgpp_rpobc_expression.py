#!/usr/bin/env python
"""Correct a degenerate ppGpp-expression fit for the rpoBC operon in a ParCa cache.

ROOT CAUSE (see the v2ecoli<->vEcoli comparison investigation): the RNAP core
complex (APORNAP-CPLX) is limited by its rpoB (EG10894) + rpoC (EG10895) subunits.
`Transcription.set_ppgpp_expression` derives each cistron's ppGpp-bound (`exp_ppgpp`)
and ppGpp-free (`exp_free`) expression ANALYTICALLY from the (correct) basal
expression + ppGpp fold-changes, then maps cistron->TU via `fit_rna_expression`, an
NNLS whose solution is known to be non-reproducible across numpy/scipy/BLAS builds
(see the comment near `adjust_polymerizing_ppgpp_expression`). For the rpoBC operon
(main TU `TU00335[c]`) this cache's NNLS landed on the non-negativity BOUNDARY:
`exp_free = 0`. That is INCONSISTENT with rpoBC's own ppGpp fold-changes
(rpoB -1.08, rpoC -0.76 => ppGpp *represses* rpoBC => exp_free must exceed exp_ppgpp).
Because `expression_from_ppgpp = f(ppGpp)*exp_ppgpp + (1-f)*exp_free`, the two splits
agree at the basal ppGpp setpoint but diverge off it: the exp_free=0 split makes
rpoBC's nutrient response run BACKWARDS off-basal (too little RNAP on rich media,
too much on poor media), which compresses assembled APORNAP-CPLX -> active_RNAP ->
rRNA/growth, reproducing the measured off-basal divergence exactly.

FIX (self-contained, basal-preserving): the rpoBC TU's total expression AT THE BASAL
ppGpp setpoint is already correct in this cache (`expression_from_ppgpp` at basal
matches the reference); only the free-vs-bound SPLIT is wrong (exp_free=0). So we keep
the basal expression `exp_free*(1-f_basal) + exp_ppgpp*f_basal` fixed and re-partition
it into a split whose ratio `exp_ppgpp/exp_free` honours the negative rpoBC ppGpp
fold-change (ppGpp represses rpoBC => exp_free > exp_ppgpp). This corrects the
off-basal nutrient response WITHOUT perturbing the (correct) basal value, and needs
no reference cache. Both vectors are then re-normalised exactly as
`_normalize_ppgpp_expression` does. Idempotent.

Usage:
    python scripts/fix_ppgpp_rpobc_expression.py <cache_dir>   [--dry-run]
    # patches <cache_dir>/simData.cPickle in place (a .bak is assumed already made)
    # then DELETE stale per-condition bundles so they regenerate from the patch:
    #   rm -rf <cache_dir>/cond_*
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp

# Fold-change-consistent free/bound RATIO exp_ppgpp/exp_free for the rpoBC operon.
# rpoBC is ppGpp-repressed (exp_free must exceed exp_ppgpp); the reference vEcoli
# ParCa's non-degenerate solve gives ratio 4.41e-5 / 1.10e-4 = 0.40 (== 2**(-1.32),
# the operon's effective ppGpp fold-change over rpoB -1.08 / rpoC -0.76). We apply
# this RATIO to THIS cache's own basal expression, so the absolute scale stays
# self-consistent (no reference-cache normalisation is imported).
RPOBC_RATIO = 0.4006   # exp_ppgpp / exp_free
RPOB_CISTRON = "EG10894_RNA"   # rpoB; its operon TU carries rpoC too


def rpobc_tu_indices(tr) -> list[int]:
    cids = [str(x) for x in tr.cistron_data["id"]]
    ci = cids.index(RPOB_CISTRON)
    M = tr.cistron_tu_mapping_matrix
    Md = M.toarray() if sp.issparse(M) else np.asarray(M)
    return sorted(int(t) for t in np.where(Md[ci] > 0)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sd_path = os.path.join(args.cache_dir, "simData.cPickle")
    with open(sd_path, "rb") as f:
        sd = pickle.load(f)
    tr = sd.process.transcription

    tus = rpobc_tu_indices(tr)
    rids = [str(x) for x in tr.rna_data["id"]]
    # The operon's expressed TU is the one currently carrying nonzero mass.
    main_tu = max(tus, key=lambda t: float(tr.exp_ppgpp[t]) + float(tr.exp_free[t]))
    pre_ppgpp = float(tr.exp_ppgpp[main_tu])
    pre_free = float(tr.exp_free[main_tu])
    print(f"rpoBC operon TUs: {[(t, rids[t]) for t in tus]}")
    print(f"main expressed TU: {main_tu} ({rids[main_tu]})")
    print(f"  BEFORE: exp_ppgpp={pre_ppgpp:.4e} exp_free={pre_free:.4e}")

    if pre_free > 0.5 * pre_ppgpp:
        print("  exp_free already fold-change-consistent (> ~exp_ppgpp); nothing to do.")
        return 0

    # ppGpp-bound fraction at the basal setpoint, from THIS cache.
    ppgpp_basal = sd.growth_rate_parameters.get_ppGpp_conc(
        sd.condition_to_doubling_time["basal"])
    f_b = float(tr.fraction_rnap_bound_ppgpp(ppgpp_basal))
    # Preserve the (correct) basal expression contribution; re-partition at the
    # fold-change-consistent ratio r = exp_ppgpp/exp_free.
    basal_val = pre_free * (1 - f_b) + pre_ppgpp * f_b
    r = RPOBC_RATIO
    free_new = basal_val / ((1 - f_b) + r * f_b)
    ppgpp_new = r * free_new
    print(f"  f_basal={f_b:.4f} basal_val={basal_val:.4e} "
          f"-> exp_ppgpp={ppgpp_new:.4e} exp_free={free_new:.4e} (ratio {r})")

    if args.dry_run:
        print("  DRY-RUN (no write).")
        return 0

    tr.exp_ppgpp[main_tu] = ppgpp_new
    tr.exp_free[main_tu] = free_new
    # Mirror Transcription._normalize_ppgpp_expression exactly.
    tr.exp_free[tr.exp_free < 0] = 0
    tr.exp_ppgpp[tr.exp_ppgpp < 0] = 0
    tr.exp_free /= tr.exp_free.sum()
    tr.exp_ppgpp /= tr.exp_ppgpp.sum()
    print(f"  AFTER : exp_ppgpp={float(tr.exp_ppgpp[main_tu]):.4e} "
          f"exp_free={float(tr.exp_free[main_tu]):.4e}")

    with open(sd_path, "wb") as f:
        pickle.dump(sd, f)
    print(f"patched {sd_path}")
    print("NOW delete stale per-condition bundles so they regenerate from the patch:")
    print(f"  rm -rf {args.cache_dir}/cond_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
