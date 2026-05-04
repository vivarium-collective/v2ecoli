#!/usr/bin/env python3
"""Run 4 scenarios sequentially, saving JSON snapshots for each."""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 2)

DUR = 2520; DT = 10
os.makedirs("out/scenarios", exist_ok=True)

def run_and_save(features, feature_configs, path, tag):
    from v2ecoli.composite import make_composite
    # Force garbage collection between runs
    import gc; gc.collect()

    print(f"[{tag}] Building ...")
    t0 = time.time()
    comp = make_composite(cache_dir="out/cache", seed=0,
                          features=features, feature_configs=feature_configs)
    print(f"[{tag}] Loaded {time.time()-t0:.1f}s, running {DUR}s ...")

    def snap(tv, cell):
        m = cell.get("listeners",{}).get("mass",{})
        s = cell.get("listeners",{}).get("sigma_factors",{})
        o = cell.get("listeners",{}).get("oxidative_stress",{})
        return {
            "time": tv,
            "dry_mass": float(m.get("dry_mass",0)),
            "cell_mass": float(m.get("cell_mass",0)),
            "protein_mass": float(m.get("protein_mass",0)),
            "rRna_mass": float(m.get("rRna_mass",0)),
            "tRna_mass": float(m.get("tRna_mass",0)),
            "mRna_mass": float(m.get("mRna_mass",0)),
            "dna_mass": float(m.get("dna_mass",0)),
            "volume": float(m.get("volume",0)),
            "growth_rate": float(m.get("instantaneous_growth_rate",0)),
            "f_sigma70": float(s.get("f_sigma70",0)),
            "f_sigma38": float(s.get("f_sigma38",0)),
            "f_sigma32": float(s.get("f_sigma32",0)),
            "f_sigma24": float(s.get("f_sigma24",0)),
            "f_sigma54": float(s.get("f_sigma54",0)),
            "Es70_count": float(s.get("Es70_count",0)),
            "EsS_count": float(s.get("EsS_count",0)),
            "ppgpp_uM": float(s.get("ppgpp_uM",0)),
            "K_E70_eff_nM": float(s.get("K_E70_eff_nM",0)),
            "phase": float(s.get("phase",0)),
            "h2o2_uM": float(o.get("h2o2_uM",0)),
            "oxyr_ox": float(o.get("oxyr_ox_fraction",0)),
            "soxr_ox": float(o.get("soxr_ox_fraction",0)),
            "oxyr_fc": float(o.get("oxyr_fold_change",1)),
            "soxrs_fc": float(o.get("soxrs_fold_change",1)),
            "total_scav": float(o.get("total_scavenging_uM_per_s",0)),
            "dna_dmg": float(o.get("dna_damage_rate",0)),
            "cum_dmg": float(o.get("cumulative_dna_damage",0)),
        }

    cell = comp.state["agents"]["0"]
    snaps = [snap(0, cell)]
    t0 = time.time(); total = 0
    while total < DUR:
        chunk = min(DT, DUR - total)
        try: comp.run(chunk)
        except: break
        total += chunk
        cell = comp.state.get("agents",{}).get("0")
        if cell is None: break
        snaps.append(snap(total, cell))
    wt = time.time() - t0
    print(f"[{tag}] {total}s in {wt:.0f}s ({total/wt:.1f}x)")

    with open(path, "w") as f:
        json.dump({"tag": tag, "features": features, "snapshots": snaps,
                   "sim_time": total, "wall_time": wt}, f)
    print(f"[{tag}] Saved {path}")
    # Free memory
    del comp, snaps
    import gc; gc.collect()

# Run whichever scenario is specified (or all)
scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

if scenario in ("1", "all"):
    run_and_save(["ppgpp_regulation"], None,
                 "out/scenarios/baseline.json", "baseline")

if scenario in ("2", "all"):
    run_and_save(["ppgpp_regulation","sigma_factor_competition"], None,
                 "out/scenarios/sigma_exp.json", "sigma_exp")

if scenario in ("3", "all"):
    run_and_save(["ppgpp_regulation","sigma_factor_competition"],
                 {"ecoli-oxidative-stress": {"external_h2o2_uM": 100.0}},
                 "out/scenarios/h2o2_stress.json", "h2o2_stress")

if scenario in ("4", "all"):
    run_and_save(["ppgpp_regulation","sigma_factor_competition","sustained_stress"],
                 {"ecoli-ppgpp-sustained-stress": {"rpos_target_count": 2000}},
                 "out/scenarios/starvation.json", "starvation")

print("\nDone!")
