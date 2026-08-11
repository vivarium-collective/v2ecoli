"""Extract per-seed lineage JSON + compute intrinsic CV of within-mother-cycle Δt.

For each seed's parquet history dir:
  - Per generation, follow the max-length agent (drops the discarded daughter).
  - Emit birth/division time+mass and every initiation event (tick where
    number_of_oric increases, with mass at that tick).
  - Save per-seed JSON.

Then aggregate across seeds:
  - Filter to mother-cycle gens (birth oriC >= 2).
  - Δt = time between the two initiation events within that cycle.
  - intrinsic CV = std(Δt) / mean(gen tau) across all pairs.

Usage:
  extract_initiation_lineage.py <parquet_history_root>... [--out-dir DIR]
"""
import argparse
import json
import os
import numpy as np
import pyarrow.dataset as ds


def extract_seed(history_root: str) -> dict:
    """Return {seed, generations: [{gen, birth_time, birth_mass, division_time,
    division_mass, tau_min, birth_oric, initiations: [{time_s_from_birth,
    cell_mass, oric_before, oric_after}]}]}."""
    d = ds.dataset(history_root, partitioning="hive")
    tbl = d.to_table(columns=[
        "generation", "global_time", "agent_id",
        "listeners__replication_data__number_of_oric",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
    ]).sort_by([("generation", "ascending"),
                ("agent_id", "ascending"),
                ("global_time", "ascending")])
    gen = tbl["generation"].to_numpy()
    tm = tbl["global_time"].to_numpy().astype(float)
    ag = tbl["agent_id"].to_numpy()
    n = tbl["listeners__replication_data__number_of_oric"].to_numpy()
    cm = tbl["listeners__mass__cell_mass"].to_numpy().astype(float)
    vol = tbl["listeners__mass__volume"].to_numpy().astype(float)

    # Infer seed from the experiment_id (parent partition)
    seed = None
    parts = history_root.rstrip("/").split("/")
    for p in parts:
        for tok in p.split("_"):
            if tok.startswith("seed"):
                try:
                    seed = int(tok.replace("seed", ""))
                except ValueError:
                    pass

    generations = []
    for g in sorted(set(gen.tolist())):
        # Pick the followed agent = the one with the most rows in this gen
        agent_ids_here = sorted(set(ag[gen == g].tolist()))
        best_a, best_n = None, -1
        for a in agent_ids_here:
            k = int(((gen == g) & (ag == a)).sum())
            if k > best_n:
                best_n = k
                best_a = a
        m = (gen == g) & (ag == best_a)
        tm_g = tm[m]
        n_g = n[m]
        cm_g = cm[m]
        vol_g = vol[m]
        if len(tm_g) < 2:
            continue

        # Initiation events: each unit of oriC increase = 1 event.
        # A single-tick 2->4 jump therefore contributes 2 events at the same
        # timestamp (both oriCs fired simultaneously, Δt = 0).
        d = np.diff(n_g)
        inits = []
        for i in np.where(d > 0)[0]:
            k = int(d[i])
            for j in range(k):
                inits.append({
                    "time_s_from_birth": float(tm_g[i + 1] - tm_g[0]),
                    "cell_mass": float(cm_g[i + 1]),
                    "cell_volume": float(vol_g[i + 1]),
                    "oric_before": int(n_g[i] + j),
                    "oric_after": int(n_g[i] + j + 1),
                })

        generations.append({
            "gen": int(g),
            "agent_id": str(best_a),
            "birth_time_s": float(tm_g[0]),
            "birth_mass": float(cm_g[0]),
            "birth_oric": int(n_g[0]),
            "division_time_s": float(tm_g[-1]),
            "division_mass": float(cm_g[-1]),
            "division_oric": int(n_g[-1]),
            "tau_min": float((tm_g[-1] - tm_g[0]) / 60.0),
            "initiations": inits,
        })

    return {"seed": seed, "history_root": history_root, "generations": generations}


def mother_cycle_deltas(lineage: dict) -> list[dict]:
    """Return one entry per mother-cycle Δt pair.

    "Mother-cycle initiation" = any initiation event firing while the cell
    already had >= 2 oriCs (i.e., oric_before >= 2). Covers 2->3, 2->4 direct,
    3->4 (second half of 2->3->4), and the "2->3->4" pattern that starts even
    in cells born at 1 oriC (after the first 1->2 firing).

    A Δt pair is formed within one generation from the FIRST TWO such events.
    Simultaneous events (both oriCs fire at the same tick) give Δt = 0."""
    out = []
    for g in lineage["generations"]:
        mother_events = [e for e in g["initiations"] if e["oric_before"] >= 2]
        if len(mother_events) < 2:
            continue
        e1, e2 = mother_events[0], mother_events[1]
        out.append({
            "seed": lineage["seed"],
            "gen": g["gen"],
            "delta_t_s": e2["time_s_from_birth"] - e1["time_s_from_birth"],
            "first_time_s": e1["time_s_from_birth"],
            "second_time_s": e2["time_s_from_birth"],
            "first_mass": e1["cell_mass"],
            "second_mass": e2["cell_mass"],
            "first_vol": e1.get("cell_volume"),
            "second_vol": e2.get("cell_volume"),
            "tau_min": g["tau_min"],
            "birth_oric": g["birth_oric"],
            "n_mother_initiations": len(mother_events),
            "n_all_initiations": len(g["initiations"]),
            "e1_before": e1.get("oric_before"),
            "e1_after":  e1.get("oric_after"),
            "e2_before": e2.get("oric_before"),
            "e2_after":  e2.get("oric_after"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history_roots", nargs="+",
                    help="Parquet history dirs (e.g. out/EXP_parquet/EXP/history/experiment_id=EXP)")
    ap.add_argument("--out-dir", default="out/synchrony_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_lineages = []
    all_pairs = []
    all_taus = []
    for root in args.history_roots:
        lineage = extract_seed(root)
        seed = lineage["seed"] if lineage["seed"] is not None else "unknown"
        json_path = os.path.join(args.out_dir, f"lineage_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(lineage, f, indent=2)
        print(f"wrote {json_path}  gens={len(lineage['generations'])}")
        all_lineages.append(lineage)
        pairs = mother_cycle_deltas(lineage)
        all_pairs.extend(pairs)
        all_taus.extend([g["tau_min"] for g in lineage["generations"]])

    if not all_pairs:
        print("no mother-cycle pairs found")
        return

    dts = np.array([p["delta_t_s"] for p in all_pairs], dtype=float)
    tau_mean_min = float(np.mean(all_taus))
    tau_mean_s = tau_mean_min * 60.0
    cv_intrinsic = float(np.std(dts, ddof=1) / tau_mean_s) if len(dts) > 1 else float("nan")

    summary = {
        "n_seeds": len(all_lineages),
        "n_pairs": len(all_pairs),
        "n_gens_total": len(all_taus),
        "tau_mean_min": tau_mean_min,
        "delta_t_stats_s": {
            "mean": float(np.mean(dts)),
            "std": float(np.std(dts, ddof=1)) if len(dts) > 1 else float("nan"),
            "min": float(np.min(dts)),
            "max": float(np.max(dts)),
            "median": float(np.median(dts)),
        },
        "intrinsic_CV": cv_intrinsic,
        "pairs": all_pairs,
    }
    out_json = os.path.join(args.out_dir, "synchrony_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== synchrony summary ===")
    print(f"seeds:        {len(all_lineages)}")
    print(f"total gens:   {len(all_taus)}   mean tau: {tau_mean_min:.1f} min")
    print(f"mother pairs: {len(all_pairs)}")
    print(f"Delta t (s):  mean {np.mean(dts):.1f}   std {np.std(dts, ddof=1) if len(dts)>1 else float('nan'):.1f}   "
          f"median {np.median(dts):.1f}   range [{np.min(dts):.1f}, {np.max(dts):.1f}]")
    print(f"intrinsic CV = std(Δt) / mean(τ) = {cv_intrinsic:.4f}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
