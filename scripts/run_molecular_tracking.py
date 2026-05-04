#!/usr/bin/env python3
"""
Run 3 scenarios tracking all key molecules from the biological flow diagrams.
Captures bulk counts for KatG, AhpCF, RpoS, ppGpp, GTP, NADPH, RNAP, etc.

Scenarios:
  A. sigma_exp       — exponential baseline (no stress)
  B. timed_h2o2      — H₂O₂ challenge at t=10 min + OxyR feedback
  C. timed_starv     — starvation at t=2 min + stringent response
"""
import os, sys, json, time, warnings, gc
import numpy as np
warnings.filterwarnings("ignore")
fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 2)
os.makedirs("out/scenarios", exist_ok=True)

DUR = 2520; DT = 10

# Bulk molecule indices (resolved at runtime)
MOL_IDS = {
    "rpod": "RPOD-MONOMER[c]",
    "rpos": "RPOS-MONOMER[c]",
    "rpoh": "RPOH-MONOMER[c]",
    "katg": "HYDROPEROXIDI-CPLX[c]",
    "kate": "HYDROPEROXIDII-CPLX[c]",
    "ahpcf": "THIOREDOXIN-REDUCT-NADPH-CPLX[c]",
    "ppgpp": "GUANOSINE-5DP-3DP[c]",
    "gtp": "GTP[c]",
    "atp": "ATP[c]",
    "nadph": "NADPH[c]",
    "h2o2": "HYDROGEN-PEROXIDE[c]",
    "rnap_free": "APORNAP-CPLX[c]",
}

_mol_idx = {}
_resolved = False

def resolve_indices(bulk):
    global _mol_idx, _resolved
    from v2ecoli.library.schema import bulk_name_to_idx
    ids = bulk["id"]
    for key, mol_id in MOL_IDS.items():
        try:
            _mol_idx[key] = bulk_name_to_idx(mol_id, ids)
        except:
            _mol_idx[key] = None
    _resolved = True

def snap(tv, cell):
    from v2ecoli.library.schema import counts
    m = cell.get("listeners",{}).get("mass",{})
    s = cell.get("listeners",{}).get("sigma_factors",{})
    o = cell.get("listeners",{}).get("oxidative_stress",{})
    ts = cell.get("listeners",{}).get("timed_stress",{})
    fb = cell.get("listeners",{}).get("oxyr_feedback",{})
    sr = cell.get("listeners",{}).get("stringent_response",{})
    bulk = cell.get("bulk")

    if not _resolved and bulk is not None:
        resolve_indices(bulk)

    d = {
        "time": tv,
        "dry_mass": float(m.get("dry_mass",0)),
        "cell_mass": float(m.get("cell_mass",0)),
        "protein_mass": float(m.get("protein_mass",0)),
        "rRna_mass": float(m.get("rRna_mass",0)),
        "tRna_mass": float(m.get("tRna_mass",0)),
        "mRna_mass": float(m.get("mRna_mass",0)),
        "growth_rate": float(m.get("instantaneous_growth_rate",0)),
        # Sigma factors
        "f_sigma70": float(s.get("f_sigma70",0)),
        "f_sigma38": float(s.get("f_sigma38",0)),
        "f_sigma32": float(s.get("f_sigma32",0)),
        "ppgpp_uM": float(s.get("ppgpp_uM",0)),
        "K_E70_eff_nM": float(s.get("K_E70_eff_nM",0)),
        "phase": float(s.get("phase",0)),
        # Oxidative stress
        "h2o2_uM": float(o.get("h2o2_uM",0)),
        "oxyr_ox": float(o.get("oxyr_ox_fraction",0)),
        "oxyr_fc": float(o.get("oxyr_fold_change",1)),
        "total_scav": float(o.get("total_scavenging_uM_per_s",0)),
        "dna_dmg": float(o.get("dna_damage_rate",0)),
        # Feedback
        "stress_active": float(ts.get("active",0)),
        "extra_katg": float(fb.get("extra_katg",0)),
        "extra_ahpcf": float(fb.get("extra_ahpcf",0)),
        # Stringent response
        "rpos_count_sr": float(sr.get("rpos_count",0)),
        "rpos_halflife": float(sr.get("rpos_halflife",0)),
    }

    # Bulk molecule counts
    if bulk is not None:
        for key, idx in _mol_idx.items():
            if idx is not None:
                d[f"bulk_{key}"] = int(counts(bulk, idx))
            else:
                d[f"bulk_{key}"] = 0

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

scenario = sys.argv[1] if len(sys.argv) > 1 else "all"
SIG = ["ppgpp_regulation", "sigma_factor_competition"]

if scenario in ("A", "all"):
    run_save(SIG + ["oxyr_feedback"], None,
             "out/scenarios/mol_baseline.json", "mol_baseline")

if scenario in ("B", "all"):
    run_save(SIG + ["timed_stress", "oxyr_feedback"],
             {"ecoli-timed-stress": {"stress_type": "h2o2", "onset_time": 600.0,
                                      "h2o2_rate_uM_per_s": 500.0}},
             "out/scenarios/mol_h2o2.json", "mol_h2o2")

if scenario in ("C", "all"):
    run_save(SIG + ["stringent_response", "oxyr_feedback"],
             {"ecoli-stringent-response": {"starvation_signal": 0.6, "onset_time": 120.0}},
             "out/scenarios/mol_starv.json", "mol_starv")

print("\nDone!")
