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


def patch_dnaa_stage1(sim_data, transcription_factor: float = 0.0,
                      te_factor: float = 0.0) -> dict:
    """Disable ParCa's DnaA translation so the constitutive Step is the
    sole DnaA source (Option A).

    Sets ``translation_efficiencies_by_monomer[PD03831] = 0`` — a clean
    per-monomer change that stops ParCa translating DnaA without touching
    the dnaN/recF operon-mates (those share dnaA's TU but not its monomer
    TE). The Stage-1 baseline recipe then adds a DnaaConstitutiveExpression
    Step that produces apo-DnaA at the absolute Stage-1 rate
    (1.5 protein/min/gene), which the equilibrium distributes to the
    ATP/ADP forms.

    ParCa still transcribes the dnaA operon mRNA; it's left untranslated
    (harmless for the DnaA level). The ``*_factor`` args are ignored —
    kept only so the (now-deprecated) calibration loop signature still
    calls cleanly; Option A needs no calibration.

    Why not patch ParCa transcription/TE to the absolute targets directly
    (Option B): empirically non-convergent — the normalised ParCa weights
    renormalise across ~4500 TUs/monomers, so scaling dnaA's weight does
    not move the realised rate monotonically. See
    scripts/calibrate_dnaa_stage1.py and feedback-2026-05-21/PLAN.md.
    """
    record: dict = {"approach": "option-A: disable ParCa DnaA translation; "
                    "constitutive Step supplies DnaA", "patched": {}}

    tl = sim_data.process.translation
    ti, tu_id, n_tus = _dnaa_tu_idx(sim_data)
    mi = _monomer_idx(sim_data)
    record["dnaA_tu"] = {"index": ti, "tu_id": tu_id, "n_tus_for_gene": n_tus,
                         "note": "operon (dnaA-dnaN-recF); transcription left intact"}
    record["monomer_index"] = mi

    te = np.array(tl.translation_efficiencies_by_monomer, copy=True)
    before = float(te[mi])
    te[mi] = 0.0
    tl.translation_efficiencies_by_monomer = te
    record["patched"]["translation_efficiency_zeroed"] = {
        "index": mi, "before": before, "after": 0.0}
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
          transcription_factor: float, te_factor: float,
          media_condition: str | None = None,
          fixed_media: str | None = None,
          c_period_minutes: float | None = None,
          d_period_seconds: float | None = None) -> None:
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

    if media_condition is not None:
        avail = dict(getattr(sim_data, "condition_to_doubling_time", {}) or {})
        if media_condition not in avail:
            raise SystemExit(
                f"unknown media_condition {media_condition!r}; known: "
                f"{sorted(avail)}")
        print(f"[{time.strftime('%H:%M:%S')}] Nutrient condition "
              f"{media_condition!r} (doubling {avail[media_condition]}), "
              f"media {fixed_media!r}")

    print(f"[{time.strftime('%H:%M:%S')}] Applying '{condition}' patch "
          f"(transcription×{transcription_factor}, te×{te_factor}) ...")
    patch_record = PATCHES[condition](
        sim_data, transcription_factor, te_factor)
    print(f"    {json.dumps(patch_record['patched'], indent=2)}")

    if c_period_minutes is not None:
        print(f"[{time.strftime('%H:%M:%S')}] C-period override: "
              f"{c_period_minutes} min (basal_elongation_rate will be "
              f"computed from replichore length)")
    if d_period_seconds is not None:
        print(f"[{time.strftime('%H:%M:%S')}] D-period override: "
              f"{d_period_seconds} s ({d_period_seconds/60} min)")

    print(f"[{time.strftime('%H:%M:%S')}] Building bundle at {cache_dir} ...")
    save_sim_input(sim_data, cache_dir,
                   condition=media_condition, fixed_media=fixed_media,
                   c_period_minutes=c_period_minutes,
                   d_period_seconds=d_period_seconds)
    write_cache_version(cache_dir, repo_root=repo_root)

    manifest = {
        "condition": condition,
        "media_condition": media_condition,
        "fixed_media": fixed_media,
        "c_period_minutes": c_period_minutes,
        "d_period_seconds": d_period_seconds,
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
    ap.add_argument("--media-condition", default=None,
                    help="ParCa nutrient condition for the initial state / "
                         "doubling time (e.g. acetate, succinate; default basal)")
    ap.add_argument("--fixed-media", default=None,
                    help="media id pinned for the whole run (e.g. minimal_acetate)")
    ap.add_argument("--c-period-min", type=float, default=None,
                    dest="c_period_minutes",
                    help="Stage-1 C-period override (e.g. 70). Forwarded to "
                         "LoadSimData; computes basal_elongation_rate from "
                         "the cached replichore length.")
    ap.add_argument("--d-period-min", type=float, default=None,
                    help="Stage-1 D-period override in MINUTES (e.g. 30). "
                         "Converted to seconds before forwarding.")
    args = ap.parse_args()
    suffix = f"-{args.media_condition}" if args.media_condition else ""
    cache_dir = args.cache_dir or f"out/cache-{args.condition}{suffix}"
    d_period_seconds = args.d_period_min * 60.0 if args.d_period_min is not None else None
    build(args.condition, args.fixture, cache_dir,
          args.transcription_factor, args.te_factor,
          media_condition=args.media_condition, fixed_media=args.fixed_media,
          c_period_minutes=args.c_period_minutes,
          d_period_seconds=d_period_seconds)


if __name__ == "__main__":
    main()
