#!/usr/bin/env python3
"""
Run complete model scenarios with all biological processes.
Saves JSON to out/scenarios/ for plotting.

8 scenarios:
  1. baseline         — original TI, no stress
  2. sigma_exp        — sigma factor competition, exponential
  3. h2o2_stress      — sigma + H₂O₂ challenge (100 µM/s from t=0)
  4. starvation       — sigma + sustained ppGpp + RpoS stabilisation
  5. timed_h2o2       — sigma + H₂O₂ at t=10min + OxyR feedback
  6. timed_starv      — sigma + stringent response at t=2min + OxyR feedback
  7. h2o2_feedback    — sigma + H₂O₂ from t=0 + OxyR feedback (shows homeostasis)
  8. combined_stress  — sigma + H₂O₂ + starvation (dual stress)
"""
import os, sys, json, time, warnings, gc
import numpy as np
warnings.filterwarnings("ignore")
fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 2)
os.makedirs("out/scenarios", exist_ok=True)

DUR = 2520; DT = 10

def snap(tv, cell):
    m = cell.get("listeners",{}).get("mass",{})
    s = cell.get("listeners",{}).get("sigma_factors",{})
    o = cell.get("listeners",{}).get("oxidative_stress",{})
    ts = cell.get("listeners",{}).get("timed_stress",{})
    fb = cell.get("listeners",{}).get("oxyr_feedback",{})
    sr = cell.get("listeners",{}).get("stringent_response",{})
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
        "oxyr_fc": float(o.get("oxyr_fold_change",1)),
        "soxrs_fc": float(o.get("soxrs_fold_change",1)),
        "total_scav": float(o.get("total_scavenging_uM_per_s",0)),
        "dna_dmg": float(o.get("dna_damage_rate",0)),
        "cum_dmg": float(o.get("cumulative_dna_damage",0)),
        "stress_active": float(ts.get("active",0)),
        "extra_katg": float(fb.get("extra_katg",0)),
        "extra_ahpcf": float(fb.get("extra_ahpcf",0)),
        "rpos_count": float(sr.get("rpos_count",0)),
        "rpos_halflife": float(sr.get("rpos_halflife",0)),
        "rela_rate": float(sr.get("rela_rate",0)),
        "spot_rate": float(sr.get("spot_rate",0)),
        "feedback_factor": float(sr.get("feedback_factor",0)),
    }

def run_save(features, fc, path, tag, dur=DUR, dt=DT):
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
SIG_FB = SIG + ["oxyr_feedback"]
SIG_SR = SIG + ["stringent_response", "oxyr_feedback"]

if scenario in ("1", "all"):
    run_save(["ppgpp_regulation"], None,
             "out/scenarios/baseline.json", "baseline")

if scenario in ("2", "all"):
    run_save(SIG, None,
             "out/scenarios/sigma_exp.json", "sigma_exp")

if scenario in ("3", "all"):
    run_save(SIG_FB,
             {"ecoli-oxidative-stress": {"external_h2o2_uM": 500.0}},
             "out/scenarios/h2o2_stress.json", "h2o2_stress")

if scenario in ("4", "all"):
    run_save(SIG + ["sustained_stress"],
             {"ecoli-ppgpp-sustained-stress": {"rpos_target_count": 2000}},
             "out/scenarios/starvation.json", "starvation")

if scenario in ("5", "all"):
    run_save(SIG + ["timed_stress", "oxyr_feedback"],
             {"ecoli-timed-stress": {"stress_type": "h2o2", "onset_time": 600.0,
                                      "h2o2_rate_uM_per_s": 500.0}},
             "out/scenarios/timed_h2o2.json", "timed_h2o2")

if scenario in ("6", "all"):
    run_save(SIG_SR,
             {"ecoli-stringent-response": {"starvation_signal": 0.6, "onset_time": 120.0}},
             "out/scenarios/timed_starv_sr.json", "timed_starv_sr")

if scenario in ("7", "all"):
    run_save(SIG_FB,
             {"ecoli-oxidative-stress": {"external_h2o2_uM": 500.0}},
             "out/scenarios/h2o2_feedback.json", "h2o2_feedback")

if scenario in ("8", "all"):
    run_save(SIG + ["timed_stress", "stringent_response", "oxyr_feedback"],
             {"ecoli-timed-stress": {"stress_type": "h2o2", "onset_time": 600.0,
                                      "h2o2_rate_uM_per_s": 500.0},
              "ecoli-stringent-response": {"starvation_signal": 0.4, "onset_time": 600.0}},
             "out/scenarios/combined_stress.json", "combined")

print("\nDone!")
