"""Build a NAMED ParCa condition cache (a tracked, reusable sim_data fit).

v2ecoli ships one ParCa cache (out/cache) hydrated from
models/parca/parca_state.pkl.gz. As we model different *conditions* (e.g.
the default ParCa fit vs. the Fu/Xiao/Jun Stage-1 heuristic parameter set),
we need each as a named, tracked, reusable cache instead of patching
sim_data ad-hoc at every composite build. This script:

  1. hydrates the live SimulationDataEcoli from the ParCa fixture (no ParCa
     re-run — same fast path as build_cache.py),
  2. applies a named PATCH to sim_data (e.g. dnaA Stage-1 transcription/TE),
  3. extracts + saves the config bundle to a named cache dir via
     save_sim_input,
  4. writes a CONDITION MANIFEST (base fixture, patch, enforced param
     targets + the scaling factors used, git sha) so the cache is
     reproducible and the param-enforcement gate (task #22) can verify it.

Generations (scripts/prepare_investigation.py) pin which condition cache
each run used; a run built on out/cache-stage1-heuristic is provably the
Stage-1 condition.

NOTE the absolute Stage-1 targets (1.5 mRNA/min/gene, 1 protein/mRNA) are
NOT closed-form in v2ecoli's normalized representation — the SCALING
FACTORS that hit them are found by scripts/calibrate_dnaa_stage1.py
(build → short sim → measure realized rate → rescale). This builder takes
the factors as inputs; default 1.0 = unpatched (use for the first
calibration measurement).

Usage:
    python scripts/build_condition_cache.py --condition stage1-heuristic \
        --transcription-factor 1.0 --te-factor 1.0
    # → out/cache-stage1-heuristic/ + its condition.json manifest
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)

DEFAULT_FIXTURE = "models/parca/parca_state.pkl.gz"

# EcoCyc ids for the dnaA gene / monomer.
DNAA_GENE_ID = "EG10235"          # gene → cistron EG10235_RNA → its TU
DNAA_MONOMER_ID = "PD03831[c]"


def _monomer_idx(sim_data, target=DNAA_MONOMER_ID):
    arr = np.asarray(sim_data.process.translation.monomer_data["id"])
    hits = np.where(arr == target)[0]
    if len(hits) == 0:
        raise KeyError(f"monomer {target!r} not found (n={len(arr)})")
    return int(hits[0])


def _dnaa_tu_idx(sim_data):
    """Map the dnaA gene (EG10235) → its transcription-unit index.

    rna_data is indexed by TU (operon), e.g. TU00259[c], not by gene. dnaA's
    cistron EG10235_RNA lives in one TU; the transcription arrays
    (basal_prob / exp_free / exp_ppgpp) are TU-indexed, so we resolve
    gene → cistron → TU. NOTE: if that TU is polycistronic (an operon),
    scaling it scales every gene in the operon — the finest granularity
    v2ecoli's TU-level transcription exposes. Flagged in the manifest.
    """
    tx = sim_data.process.transcription
    gene_ids = np.asarray(tx.cistron_data["gene_id"])
    ci = np.where(gene_ids == DNAA_GENE_ID)[0]
    if len(ci) == 0:
        raise KeyError(f"gene {DNAA_GENE_ID!r} not in cistron_data")
    m = tx.cistron_tu_mapping_matrix
    row = (np.asarray(m[ci[0]].todense()).ravel()
           if hasattr(m, "todense") else np.asarray(m)[ci[0]])
    tus = np.where(row != 0)[0]
    if len(tus) == 0:
        raise KeyError(f"no TU contains cistron for {DNAA_GENE_ID!r}")
    rna_ids = np.asarray(tx.rna_data["id"])
    return int(tus[0]), str(rna_ids[tus[0]]), int(len(tus))


def patch_dnaa_stage1(sim_data, transcription_factor: float,
                      te_factor: float) -> dict:
    """Scale dnaA transcription + translation knobs by the given factors.

    Patches BOTH the ppGpp-on path (exp_free / exp_ppgpp) and the ppGpp-off
    path (basal_prob) for EG10235_RNA, and the translation-efficiency weight
    for PD03831. Returns a record of what changed (for the manifest). The
    absolute Stage-1 targets are realised by the calibration loop choosing
    the factors; this function just applies them.
    """
    record: dict = {"transcription_factor": transcription_factor,
                    "te_factor": te_factor, "patched": {}}

    tx = sim_data.process.transcription
    treg = sim_data.process.transcription_regulation
    tl = sim_data.process.translation

    ti, tu_id, n_tus = _dnaa_tu_idx(sim_data)
    mi = _monomer_idx(sim_data)
    record["dnaA_tu"] = {"index": ti, "tu_id": tu_id,
                         "n_tus_for_gene": n_tus}
    record["monomer_index"] = mi

    # Transcription: scale every array that feeds the realized synthesis
    # rate, so the patch holds whether or not ppGpp regulation is active.
    for attr, owner in (("exp_free", tx), ("exp_ppgpp", tx),
                        ("basal_prob", treg)):
        a = getattr(owner, attr, None)
        if a is not None and len(a) > ti:
            a = np.array(a, copy=True)
            before = float(a[ti])
            a[ti] = before * transcription_factor
            setattr(owner, attr, a)
            record["patched"][attr] = {"index": ti, "before": before,
                                       "after": float(a[ti])}

    # Translation efficiency weight for dnaA monomer.
    te = np.array(tl.translation_efficiencies_by_monomer, copy=True)
    before = float(te[mi])
    te[mi] = before * te_factor
    tl.translation_efficiencies_by_monomer = te
    record["patched"]["translation_efficiency"] = {
        "index": mi, "before": before, "after": float(te[mi])}

    return record


PATCHES = {
    "stage1-heuristic": patch_dnaa_stage1,
}

# Absolute Stage-1 targets recorded in the manifest (the values the
# calibration loop aims the scaling factors at).
TARGETS = {
    "stage1-heuristic": {
        "dnaA_transcription_rate_mrna_per_min_per_gene": 1.5,
        "dnaA_translation_efficiency_protein_per_mrna": 1.0,
        "source": "wcm_stage1_parameters (Fu/Xiao/Jun 2023)",
    },
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(condition: str, fixture: str, cache_dir: str,
          transcription_factor: float, te_factor: float) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    if condition not in PATCHES:
        raise SystemExit(f"unknown condition {condition!r}; "
                         f"known: {sorted(PATCHES)}")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading fixture {fixture} ...")
    state = load_parca_state(fixture)
    print(f"[{time.strftime('%H:%M:%S')}] Hydrating sim_data ...")
    sim_data = hydrate_sim_data_from_state(state)

    print(f"[{time.strftime('%H:%M:%S')}] Applying '{condition}' patch "
          f"(transcription×{transcription_factor}, te×{te_factor}) ...")
    patch_record = PATCHES[condition](
        sim_data, transcription_factor, te_factor)
    print(f"    {json.dumps(patch_record['patched'], indent=2)}")

    print(f"[{time.strftime('%H:%M:%S')}] Building bundle at {cache_dir} ...")
    save_sim_input(sim_data, cache_dir)
    write_cache_version(cache_dir, repo_root=repo_root)

    manifest = {
        "condition": condition,
        "created_at": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "base_fixture": fixture,
        "targets": TARGETS.get(condition, {}),
        "patch": patch_record,
    }
    with open(os.path.join(cache_dir, "condition.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    condition manifest → {cache_dir}/condition.json")
    print(f"\nTotal: {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="stage1-heuristic",
                    help=f"condition name (known: {sorted(PATCHES)})")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--cache", dest="cache_dir", default=None,
                    help="output dir (default: out/cache-<condition>)")
    ap.add_argument("--transcription-factor", type=float, default=1.0,
                    help="scale dnaA transcription arrays (calibration finds this)")
    ap.add_argument("--te-factor", type=float, default=1.0,
                    help="scale dnaA translation-efficiency weight")
    args = ap.parse_args()
    cache_dir = args.cache_dir or f"out/cache-{args.condition}"
    build(args.condition, args.fixture, cache_dir,
          args.transcription_factor, args.te_factor)


if __name__ == "__main__":
    main()
