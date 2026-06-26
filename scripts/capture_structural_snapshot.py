"""Regenerate v2ecoli/structural/data/v2ecoli_state*.npz from a live simulation.

Runs the baseline composite for a short advance (default 2 s) to capture the
birth-state bulk counts, cell volume, chromosome state, and active-RNAP loci.
Saves the result to v2ecoli/structural/data/v2ecoli_state.npz so that
build.py / rnap_state() can read it without re-running the simulation.

If a cache is available at out/cache (or the path given by --cache-dir), the
pre-division snapshot (v2ecoli_state_division.npz) is also regenerated from a
two-generation multigen run at the end of generation 1.

Usage::

    cd /path/to/v2e-3d-txn  # CWD must be the worktree so v2ecoli imports correctly
    .venv/bin/python scripts/capture_structural_snapshot.py [--cache-dir out/cache]

Requires: a valid ParCa cache (see reference_v2ecoli_worktree_cache_symlink).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Genome constants (must match build.py)
# ---------------------------------------------------------------------------
GENOME_BP = 4_641_652
REPLICHORE_BP = GENOME_BP // 2  # 2,320,826

DATA = Path(__file__).parent.parent / "v2ecoli" / "structural" / "data"


# ---------------------------------------------------------------------------
# Helpers: extract per-cell snapshot from composite state
# ---------------------------------------------------------------------------

def _magnitude(val, default=0.0):
    """Strip pint Quantity, return float."""
    return float(getattr(val, "magnitude", val)) if val is not None else default


def _extract_snapshot(comp):
    """Extract all snapshot fields from a running composite.

    Returns a dict with keys matching the npz schema:
      ids, counts, volume, n_chromosomes, fork_fraction, division_progress,
      rnap_coordinates, rnap_domain_index, rnap_is_forward
    """
    cell = comp.state.get("agents", {}).get("0", comp.state)

    # ── Bulk molecules ───────────────────────────────────────────────────────
    bulk = cell["bulk"]
    ids = np.array([str(x) for x in bulk["id"]])
    counts = np.array(bulk["count"], dtype=np.int64)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_raw = cell.get("listeners", {}).get("mass", {}).get("volume")
    volume = float(_magnitude(vol_raw, 1.0))

    # ── Chromosome state ─────────────────────────────────────────────────────
    unique = cell.get("unique", {})

    # Number of full chromosomes
    fc = unique.get("full_chromosome")
    n_chromosomes = 0
    division_time_s = 0.0
    if fc is not None and hasattr(fc, "dtype") and "_entryState" in fc.dtype.names:
        active_fc = fc[fc["_entryState"].view(bool)]
        n_chromosomes = int(len(active_fc))
        if "division_time" in fc.dtype.names and len(active_fc) > 0:
            # division_time is the absolute sim-time at which division fires
            division_time_s = float(active_fc["division_time"].max())

    # Fork fraction: mean distance of active replication forks from oriC,
    # expressed as a fraction of the replichore length.
    rep = unique.get("active_replisome")
    fork_fraction = 0.0
    if rep is not None and hasattr(rep, "dtype") and "_entryState" in rep.dtype.names:
        active_rep = rep[rep["_entryState"].view(bool)]
        if len(active_rep) > 0 and "coordinates" in rep.dtype.names:
            fork_fraction = float(np.mean(np.abs(active_rep["coordinates"]))) / REPLICHORE_BP

    # Division progress: fraction of D-period elapsed (0 = newborn, 1 = dividing)
    global_time = float(cell.get("global_time", 0.0))
    D_PERIOD_S = 27.0 * 60  # ~27 minutes, standard E. coli D-period
    if division_time_s > 0:
        # D-period started at (division_time - D_PERIOD_S); progress = elapsed/D_PERIOD_S
        d_start = division_time_s - D_PERIOD_S
        division_progress = float(np.clip((global_time - d_start) / D_PERIOD_S, 0.0, 1.0))
    else:
        division_progress = 0.0

    # ── Active RNAP ──────────────────────────────────────────────────────────
    rnap = unique.get("active_RNAP")
    rnap_coordinates = np.array([], dtype=np.int64)
    rnap_domain_index = np.array([], dtype=np.int32)
    rnap_is_forward = np.array([], dtype=bool)

    rnap_unique_index = np.array([], dtype=np.int64)

    if rnap is not None and hasattr(rnap, "dtype") and "_entryState" in rnap.dtype.names:
        active_rnap = rnap[rnap["_entryState"].view(bool)]
        n_rnap = len(active_rnap)
        if n_rnap > 0:
            if "coordinates" in rnap.dtype.names:
                rnap_coordinates = active_rnap["coordinates"].astype(np.int64)
            if "domain_index" in rnap.dtype.names:
                rnap_domain_index = active_rnap["domain_index"].astype(np.int32)
            if "is_forward" in rnap.dtype.names:
                rnap_is_forward = active_rnap["is_forward"].astype(bool)
            elif "direction" in rnap.dtype.names:
                # Older schema: direction is +1/-1 or True/False
                rnap_is_forward = (active_rnap["direction"] > 0).astype(bool)
            if "unique_index" in rnap.dtype.names:
                rnap_unique_index = active_rnap["unique_index"].astype(np.int64)
        print(f"  Captured {n_rnap} active RNAPs  "
              f"(coordinates range: "
              f"{int(rnap_coordinates.min()) if n_rnap else 'n/a'} .. "
              f"{int(rnap_coordinates.max()) if n_rnap else 'n/a'} bp)")
    else:
        print("  WARNING: active_RNAP unique molecule not found in state")

    # ── Nascent RNA ──────────────────────────────────────────────────────────
    rna = unique.get("RNA")
    rna_unique_index = np.array([], dtype=np.int64)
    rna_RNAP_index = np.array([], dtype=np.int64)
    rna_transcript_length = np.array([], dtype=np.int64)
    rna_is_mRNA = np.array([], dtype=bool)
    rna_is_full_transcript = np.array([], dtype=bool)
    rna_TU_index = np.array([], dtype=np.int64)

    if rna is not None and hasattr(rna, "dtype") and "_entryState" in rna.dtype.names:
        active_rna = rna[rna["_entryState"].view(bool)]
        n_rna = len(active_rna)
        if n_rna > 0:
            if "unique_index" in rna.dtype.names:
                rna_unique_index = active_rna["unique_index"].astype(np.int64)
            if "RNAP_index" in rna.dtype.names:
                rna_RNAP_index = active_rna["RNAP_index"].astype(np.int64)
            if "transcript_length" in rna.dtype.names:
                rna_transcript_length = active_rna["transcript_length"].astype(np.int64)
            if "is_mRNA" in rna.dtype.names:
                rna_is_mRNA = active_rna["is_mRNA"].astype(bool)
            if "is_full_transcript" in rna.dtype.names:
                rna_is_full_transcript = active_rna["is_full_transcript"].astype(bool)
            if "TU_index" in rna.dtype.names:
                rna_TU_index = active_rna["TU_index"].astype(np.int64)
        print(f"  Captured {n_rna} nascent RNAs  "
              f"(mRNA={int(rna_is_mRNA.sum())}, "
              f"full_transcript={int(rna_is_full_transcript.sum())})")
    else:
        print("  WARNING: RNA unique molecule not found in state")

    return {
        "ids": ids,
        "counts": counts,
        "volume": np.float64(volume),
        "n_chromosomes": np.int64(n_chromosomes),
        "fork_fraction": np.float64(fork_fraction),
        "division_progress": np.float64(division_progress),
        "rnap_coordinates": rnap_coordinates,
        "rnap_domain_index": rnap_domain_index,
        "rnap_is_forward": rnap_is_forward,
        "rnap_unique_index": rnap_unique_index,
        "rna_unique_index": rna_unique_index,
        "rna_RNAP_index": rna_RNAP_index,
        "rna_transcript_length": rna_transcript_length,
        "rna_is_mRNA": rna_is_mRNA,
        "rna_is_full_transcript": rna_is_full_transcript,
        "rna_TU_index": rna_TU_index,
    }


# ---------------------------------------------------------------------------
# Main capture logic
# ---------------------------------------------------------------------------

def capture_snapshot(cache_dir: str = "out/cache", advance_s: float = 2.0,
                     seed: int = 0, skip_division: bool = False):
    """Build and run the baseline composite; save snapshots to the data dir."""
    import v2ecoli

    DATA.mkdir(parents=True, exist_ok=True)

    # ── Birth-state snapshot (short run) ────────────────────────────────────
    print(f"[capture] Building baseline composite (seed={seed}, "
          f"cache_dir={cache_dir!r}) ...")
    comp = v2ecoli.build_composite("baseline", seed=seed, cache_dir=cache_dir)
    print(f"[capture] Running {advance_s} s ...")
    comp.run(advance_s)

    print("[capture] Extracting birth-state snapshot ...")
    snap = _extract_snapshot(comp)

    out_birth = DATA / "v2ecoli_state.npz"
    np.savez(out_birth, **snap)
    print(f"[capture] Saved birth snapshot → {out_birth}")
    print(f"          ids={len(snap['ids'])}, volume={snap['volume']:.3f} fL, "
          f"n_chromosomes={snap['n_chromosomes']}, "
          f"fork_fraction={snap['fork_fraction']:.4f}, "
          f"division_progress={snap['division_progress']:.4f}")
    print(f"          RNAP count = {len(snap['rnap_coordinates'])}, "
          f"RNA count = {len(snap['rna_unique_index'])}")

    if skip_division:
        print("[capture] Skipping division snapshot (--skip-division).")
        return

    # ── Division-state snapshot (long single-cell run to pre-division) ───────
    print("\n[capture] Building pre-division snapshot via long single-cell run ...")
    try:
        # Run in 100 s chunks, stop once division is imminent
        # (global_time ≈ division_time - 10 s)
        print("  Running up to ~3000 s of sim time to capture pre-division state ...")
        comp2 = v2ecoli.build_composite("baseline", seed=seed, cache_dir=cache_dir)
        # Stop the simulation safely before division fires.
        # Division happens at division_time; we run in 100 s chunks, so we
        # stop once we're within 110 s of division_time (one chunk margin).
        div_t_known = 0.0
        for chunk_i in range(35):
            cell2 = comp2.state.get("agents", {}).get("0", comp2.state)
            t = float(cell2.get("global_time", 0.0))
            fc2 = cell2.get("unique", {}).get("full_chromosome")
            if fc2 is not None and hasattr(fc2, "dtype") and "division_time" in fc2.dtype.names:
                active_fc2 = fc2[fc2["_entryState"].view(bool)]
                if len(active_fc2) > 0:
                    div_t = float(active_fc2["division_time"].max())
                    if div_t > 0:
                        div_t_known = div_t
                    if div_t_known > 0 and t >= div_t_known - 110.0:
                        print(f"  Stopping at t={t:.1f} s (division_time={div_t_known:.1f} s, "
                              f"within 110 s margin)")
                        break
                    print(f"  t={t:.0f} s (division_time={div_t_known:.0f} s, "
                          f"{max(0.0, div_t_known - t):.0f} s remaining) ...")
                else:
                    print(f"  t={t:.0f} s ...")
            else:
                print(f"  t={t:.0f} s ...")
            comp2.run(100.0)
        else:
            print("  WARNING: ran 3500 s without stopping — using final state")

        snap_div = _extract_snapshot(comp2)
        # Override division_progress to 1.0 for the pre-division frame
        snap_div["division_progress"] = np.float64(1.0)

        out_div = DATA / "v2ecoli_state_division.npz"
        np.savez(out_div, **snap_div)
        print(f"[capture] Saved division snapshot → {out_div}")
        print(f"          volume={snap_div['volume']:.3f} fL, "
              f"n_chromosomes={snap_div['n_chromosomes']}, "
              f"fork_fraction={snap_div['fork_fraction']:.4f}")
        print(f"          RNAP count = {len(snap_div['rnap_coordinates'])}, "
              f"RNA count = {len(snap_div['rna_unique_index'])}")

    except Exception as exc:
        print(f"[capture] Division snapshot FAILED: {exc!r}")
        print("          The birth snapshot was saved; run separately on the mini.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="out/cache",
                        help="ParCa cache directory (default: out/cache)")
    parser.add_argument("--advance-s", type=float, default=2.0,
                        help="Simulation time for the birth snapshot (default: 2.0 s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--skip-division", action="store_true",
                        help="Skip the slow 2-gen division-state run")
    args = parser.parse_args()
    capture_snapshot(
        cache_dir=args.cache_dir,
        advance_s=args.advance_s,
        seed=args.seed,
        skip_division=args.skip_division,
    )


if __name__ == "__main__":
    main()
