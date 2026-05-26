"""Calibrate the dnaA Stage-1 condition cache to the absolute expert targets.

The Stage-1 targets (1.5 dnaA mRNA/min/gene, TE 1 protein/mRNA) are NOT
closed-form in v2ecoli's normalized ParCa representation — basal_prob /
expression / TE weights renormalize across all TUs/monomers. So we calibrate
empirically: build the condition cache with scaling factors, run a short sim,
measure the REALIZED rates from listeners, rescale the factors by
target/realized, and repeat until within tolerance.

Measured quantities (per iteration):
  - dnaA mRNA synthesis rate (mRNA/min/gene)
      = sum_t rnap_data.rna_init_event_per_cistron[dnaA_cistron]
        / run_minutes / mean(gene_copy_number[dnaA])
  - realized TE (protein/mRNA)
      = total dnaA ribosome-initiation events / total dnaA mRNA synthesized
        over the run. Both are direct INITIATION-EVENT counts
        (ribosome_data.ribosome_init_event_per_monomer[dnaA] and
        rnap_data.rna_init_event_per_cistron[dnaA]), so this is the true
        proteins-made-per-mRNA-made ratio — NOT a count-delta (free-monomer
        count is confounded by ATP/ADP binding + degradation, which is why
        the earlier Δprotein proxy gave a spurious 0.037).

Rescale rule (locally linear away from the renorm denominator):
  transcription_factor *= target_mrna_rate / realized_mrna_rate
  te_factor            *= target_te        / realized_te

Usage:
    python scripts/calibrate_dnaa_stage1.py --iterations 1 --sim-minutes 15
    # iteration 0 uses factors 1.0 (baseline measurement)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DNAA_CISTRON_GENE = "EG10235"
DNAA_MONOMER_IDX = 3861          # PD03831[c] in monomer_counts (confirmed)
TARGET_MRNA_RATE = 1.5           # mRNA/min/gene
TARGET_TE = 1.0                  # protein/mRNA


def _dnaa_cistron_index(sim_data) -> int:
    gene_ids = np.asarray(sim_data.process.transcription.cistron_data["gene_id"])
    return int(np.where(gene_ids == DNAA_CISTRON_GENE)[0][0])


def measure(cache_dir: str, sim_minutes: float, seed: int = 0) -> dict:
    """Run the Stage-1 baseline on `cache_dir` and measure realized rates."""
    import warnings
    warnings.filterwarnings("ignore")
    from v2ecoli.composites.baseline_recipes import dnaa_00_baseline_with_dnaa_readout
    from process_bigraph import Composite
    import v2ecoli.core as vc
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)

    cistron_idx = _dnaa_cistron_index(
        hydrate_sim_data_from_state(load_parca_state("models/parca/parca_state.pkl.gz")))

    # CRITICAL: v2ecoli.core._load_cache_bundle_cached is lru_cache'd by
    # cache_dir, so a cache rebuilt between iterations is NOT reloaded within
    # one process. Clear it so each measurement reflects the freshly-built
    # cache. (Without this, iteration 1+ silently measures the iteration-0
    # bundle.) Best practice: also run each iteration in a fresh subprocess.
    vc._load_cache_bundle_cached.cache_clear()

    core = vc.build_core()
    doc = dnaa_00_baseline_with_dnaa_readout(seed=seed, cache_dir=cache_dir)
    comp = Composite({"state": doc["state"]}, core=core)

    total_s = sim_minutes * 60.0
    chunk = 60.0
    done = 0.0
    mrna_events = 0.0          # dnaA mRNA initiation events (RNAP)
    protein_events = 0.0       # dnaA ribosome initiation events
    copy_samples: list[float] = []

    def _read(path):
        node = comp.state["agents"]["0"].get("listeners") or {}
        for k in path:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        return node

    while done < total_s:
        step = min(chunk, total_s - done)
        comp.run(step)
        done += step
        ev = _read(["rnap_data", "rna_init_event_per_cistron"])
        if ev is not None and len(ev) > cistron_idx:
            mrna_events += float(ev[cistron_idx])
        ri = _read(["ribosome_data", "ribosome_init_event_per_monomer"])
        if ri is not None and len(ri) > DNAA_MONOMER_IDX:
            protein_events += float(ri[DNAA_MONOMER_IDX])
        cn = _read(["rna_synth_prob", "gene_copy_number"])
        if cn is not None and len(cn) > cistron_idx:
            copy_samples.append(float(cn[cistron_idx]))

    run_min = total_s / 60.0
    mean_copy = float(np.mean(copy_samples)) if copy_samples else 1.0
    mrna_rate = mrna_events / run_min / max(mean_copy, 1e-9)         # mRNA/min/gene
    # True proteins-made-per-mRNA-made (both are initiation-event counts).
    realized_te = (protein_events / mrna_events) if mrna_events > 0 else float("nan")
    return {
        "mrna_events": mrna_events,
        "protein_events": protein_events,
        "mean_gene_copy": mean_copy,
        "mrna_rate_per_min_per_gene": mrna_rate,
        "realized_te_protein_per_mrna": realized_te,
        "run_minutes": run_min,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--sim-minutes", type=float, default=15.0)
    ap.add_argument("--tol", type=float, default=0.15, help="rel. tolerance")
    ap.add_argument("--condition", default="stage1-heuristic")
    ap.add_argument("--media-condition", default=None,
                    help="ParCa nutrient condition (e.g. glycerol). Forwarded "
                         "to build_condition_cache.py. Cache dir gets the "
                         "matching -{media} suffix.")
    ap.add_argument("--fixed-media", default=None,
                    help="media id (e.g. minimal_glycerol). Forwarded.")
    ap.add_argument("--fixture", default=None,
                    help="parca_state.pkl.gz path; pass when calibrating "
                         "against a fresh-rebuild fixture (e.g. glycerol).")
    args = ap.parse_args()

    import subprocess
    tx_factor, te_factor = 1.0, 1.0
    suffix = f"-{args.media_condition}" if args.media_condition else ""
    cache_dir = f"out/cache-{args.condition}{suffix}"
    history = []
    for it in range(args.iterations + 1):  # iteration 0 = baseline measurement
        print(f"\n=== iteration {it}: tx_factor={tx_factor:.4g} te_factor={te_factor:.4g} ===",
              flush=True)
        t0 = time.time()
        build_cmd = [sys.executable, "scripts/build_condition_cache.py",
                     "--condition", args.condition,
                     "--transcription-factor", str(tx_factor),
                     "--te-factor", str(te_factor)]
        if args.media_condition:
            build_cmd += ["--media-condition", args.media_condition]
        if args.fixed_media:
            build_cmd += ["--fixed-media", args.fixed_media]
        if args.fixture:
            build_cmd += ["--fixture", args.fixture]
        subprocess.run(build_cmd, check=True, capture_output=True)
        m = measure(cache_dir, args.sim_minutes)
        m.update({"iteration": it, "tx_factor": tx_factor, "te_factor": te_factor,
                  "wall_s": round(time.time() - t0, 1)})
        history.append(m)
        print(f"  realized: {m['mrna_rate_per_min_per_gene']:.4g} mRNA/min/gene "
              f"(target {TARGET_MRNA_RATE}), TE {m['realized_te_protein_per_mrna']:.4g} "
              f"(target {TARGET_TE}); mean_copy={m['mean_gene_copy']:.3g}", flush=True)
        # Rescale for next iteration.
        rate = m["mrna_rate_per_min_per_gene"]
        te = m["realized_te_protein_per_mrna"]
        if rate and rate > 0:
            tx_factor *= TARGET_MRNA_RATE / rate
        if te and te == te and te > 0:  # not NaN
            te_factor *= TARGET_TE / te
        within = (abs(rate - TARGET_MRNA_RATE) / TARGET_MRNA_RATE < args.tol
                  if rate else False)
        if within and it > 0:
            print("  → within tolerance, stopping.")
            break

    print("\n=== CALIBRATION HISTORY ===")
    print(json.dumps(history, indent=2))
    print(f"\nSuggested factors for next build: "
          f"--transcription-factor {tx_factor:.6g} --te-factor {te_factor:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
