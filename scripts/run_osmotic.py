#!/usr/bin/env python3
"""Run osmotic stress scenarios with molecular tracking."""
import os, sys, json, time, warnings, gc
import numpy as np
warnings.filterwarnings("ignore")
fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 2)
os.makedirs("out/scenarios", exist_ok=True)

DUR = 2520; DT = 10

from v2ecoli.library.schema import counts, bulk_name_to_idx

MOL_IDS = {
    "rpod": "RPOD-MONOMER[c]", "rpos": "RPOS-MONOMER[c]",
    "ppgpp": "GUANOSINE-5DP-3DP[c]", "gtp": "GTP[c]",
    "katg": "HYDROPEROXIDI-CPLX[c]", "ahpcf": "THIOREDOXIN-REDUCT-NADPH-CPLX[c]",
    "rnap_free": "APORNAP-CPLX[c]",
}
_mol_idx = {}; _resolved = False

def resolve(bulk):
    global _mol_idx, _resolved
    ids = bulk["id"]
    for k, mid in MOL_IDS.items():
        try: _mol_idx[k] = bulk_name_to_idx(mid, ids)
        except: _mol_idx[k] = None
    _resolved = True

def snap(tv, cell):
    m = cell.get("listeners",{}).get("mass",{})
    s = cell.get("listeners",{}).get("sigma_factors",{})
    o = cell.get("listeners",{}).get("oxidative_stress",{})
    osm = cell.get("listeners",{}).get("osmotic_stress",{})
    fb = cell.get("listeners",{}).get("oxyr_feedback",{})
    bulk = cell.get("bulk")
    if not _resolved and bulk is not None: resolve(bulk)
    d = {
        "time": tv,
        "dry_mass": float(m.get("dry_mass",0)),
        "cell_mass": float(m.get("cell_mass",0)),
        "protein_mass": float(m.get("protein_mass",0)),
        "rRna_mass": float(m.get("rRna_mass",0)),
        "tRna_mass": float(m.get("tRna_mass",0)),
        "mRna_mass": float(m.get("mRna_mass",0)),
        "growth_rate": float(m.get("instantaneous_growth_rate",0)),
        "f_sigma70": float(s.get("f_sigma70",0)),
        "f_sigma38": float(s.get("f_sigma38",0)),
        "f_sigma32": float(s.get("f_sigma32",0)),
        "ppgpp_uM": float(s.get("ppgpp_uM",0)),
        "K_E70_eff_nM": float(s.get("K_E70_eff_nM",0)),
        "phase": float(s.get("phase",0)),
        "h2o2_uM": float(o.get("h2o2_uM",0)),
        "oxyr_fc": float(o.get("oxyr_fold_change",1)),
        # Osmotic stress specific
        "osmolarity": float(osm.get("osmolarity",0.28)),
        "turgor_atm": float(osm.get("turgor_atm",1.0)),
        "volume_fraction": float(osm.get("volume_fraction",1.0)),
        "ompr_p_fraction": float(osm.get("ompr_p_fraction",0)),
        "k_plus_mM": float(osm.get("k_plus_mM",200)),
        "glutamate_mM": float(osm.get("glutamate_mM",30)),
        "trehalose_mM": float(osm.get("trehalose_mM",0)),
        "growth_inhibition": float(osm.get("growth_inhibition",0)),
        "stress_active": float(osm.get("stress_active",0)),
        "porin_ompC_up": float(osm.get("porin_ompC_up",0)),
        "porin_ompF_down": float(osm.get("porin_ompF_down",0)),
    }
    if bulk is not None:
        for k, idx in _mol_idx.items():
            d[f"bulk_{k}"] = int(counts(bulk, idx)) if idx is not None else 0
    return d

def run_save(features, fc, path, tag, dur=DUR, dt=DT):
    global _resolved, _mol_idx
    _resolved = False; _mol_idx = {}
    from v2ecoli.composite import make_composite
    gc.collect()
    print(f"[{tag}] Building ..."); t0=time.time()
    comp = make_composite(cache_dir="out/cache", seed=0,
                          features=features, feature_configs=fc)
    print(f"[{tag}] Loaded {time.time()-t0:.1f}s, running {dur}s ...")
    cell = comp.state["agents"]["0"]
    snaps = [snap(0, cell)]
    t0=time.time(); total=0
    while total < dur:
        chunk = min(dt, dur-total)
        try: comp.run(chunk)
        except: break
        total += chunk
        cell = comp.state.get("agents",{}).get("0")
        if cell is None: break
        snaps.append(snap(total, cell))
    wt=time.time()-t0
    print(f"[{tag}] {total}s in {wt:.0f}s ({total/wt:.1f}x)")
    with open(path,"w") as f:
        json.dump({"tag":tag,"features":features,"snapshots":snaps,
                   "sim_time":total,"wall_time":wt},f)
    print(f"[{tag}] Saved {path}")
    del comp; gc.collect()

SIG = ["ppgpp_regulation", "sigma_factor_competition"]

scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

# Exponential baseline (no osmotic stress)
if scenario in ("1", "all"):
    run_save(SIG + ["osmotic_stress", "oxyr_feedback"], None,
             "out/scenarios/osm_baseline.json", "osm_baseline")

# Moderate osmotic stress: 0.6 Osm at t=10 min
if scenario in ("2", "all"):
    run_save(SIG + ["osmotic_stress", "oxyr_feedback"],
             {"ecoli-osmotic-stress": {"stress_osmolarity": 0.6, "onset_time": 600.0}},
             "out/scenarios/osm_moderate.json", "osm_moderate")

# Severe osmotic stress: 1.0 Osm at t=10 min
if scenario in ("3", "all"):
    run_save(SIG + ["osmotic_stress", "oxyr_feedback"],
             {"ecoli-osmotic-stress": {"stress_osmolarity": 1.0, "onset_time": 600.0}},
             "out/scenarios/osm_severe.json", "osm_severe")

print("\nDone!")
