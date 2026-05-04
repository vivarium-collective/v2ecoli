#!/usr/bin/env python3
"""
Run advanced scenarios: timed stress onset, OxyR feedback, multi-generation.
Saves JSON snapshots to out/scenarios/ for plotting.

Scenarios:
  5. timed_h2o2      — H₂O₂ challenge at t=600s (10 min) with OxyR feedback
  6. timed_starvation — ppGpp starvation at t=600s with OxyR feedback
  7. multigen_starv   — starvation across cell division (follows daughter)
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
    return {
        "time": tv, "dry_mass": float(m.get("dry_mass",0)),
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

def run_multigen(features, fc, path, tag, n_gen=2, dur_per_gen=2520, dt=10):
    """Run simulation, follow first daughter through division, repeat."""
    from v2ecoli.composite import make_composite
    gc.collect()
    print(f"[{tag}] Building (multigen, {n_gen} generations) ...")
    t0_total = time.time()
    comp = make_composite(cache_dir="out/cache", seed=0,
                          features=features, feature_configs=fc)
    all_snaps = []
    cumulative_time = 0.0
    agent_id = "0"

    for gen in range(n_gen):
        print(f"[{tag}] Generation {gen+1}/{n_gen}, agent={agent_id} ...")
        t0 = time.time()
        gen_time = 0
        while gen_time < dur_per_gen:
            chunk = min(dt, dur_per_gen - gen_time)
            try:
                comp.run(chunk)
            except Exception:
                break
            gen_time += chunk
            cumulative_time += chunk

            cell = comp.state.get("agents",{}).get(agent_id)
            if cell is None:
                # Division happened — find daughter
                agents = comp.state.get("agents",{})
                daughters = [k for k in agents.keys() if k != agent_id and k.startswith(agent_id)]
                if not daughters:
                    daughters = list(agents.keys())
                if daughters:
                    agent_id = sorted(daughters)[0]
                    cell = agents[agent_id]
                    print(f"[{tag}]   Division at t={cumulative_time:.0f}s, following {agent_id}")
                else:
                    print(f"[{tag}]   Division at t={cumulative_time:.0f}s, no daughters found")
                    break

            if cell is not None:
                s = snap(cumulative_time, cell)
                s["generation"] = gen + 1
                s["agent_id"] = agent_id
                all_snaps.append(s)

        wt = time.time() - t0
        print(f"[{tag}]   Gen {gen+1}: {gen_time}s in {wt:.0f}s")

    total_wt = time.time() - t0_total
    print(f"[{tag}] Total: {cumulative_time:.0f}s sim in {total_wt:.0f}s wall")
    with open(path, "w") as f:
        json.dump({"tag": tag, "features": features, "snapshots": all_snaps,
                   "sim_time": cumulative_time, "wall_time": total_wt,
                   "n_generations": n_gen}, f)
    print(f"[{tag}] Saved {path}")
    del comp; gc.collect()

# ── Run scenarios ─────────────────────────────────────────────────────────
scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

SIGMA_FEATS = ["ppgpp_regulation", "sigma_factor_competition"]

if scenario in ("5", "all"):
    run_save(
        SIGMA_FEATS + ["timed_stress", "oxyr_feedback"],
        {"ecoli-timed-stress": {"stress_type": "h2o2", "onset_time": 600.0,
                                 "h2o2_rate_uM_per_s": 100.0}},
        "out/scenarios/timed_h2o2.json", "timed_h2o2")

if scenario in ("6", "all"):
    run_save(
        SIGMA_FEATS + ["timed_stress", "oxyr_feedback"],
        {"ecoli-timed-stress": {"stress_type": "starvation", "onset_time": 600.0,
                                 "ppgpp_target": 250000}},
        "out/scenarios/timed_starvation.json", "timed_starv")

if scenario in ("7", "all"):
    run_multigen(
        SIGMA_FEATS + ["timed_stress", "oxyr_feedback"],
        {"ecoli-timed-stress": {"stress_type": "starvation", "onset_time": 300.0,
                                 "ppgpp_target": 250000}},
        "out/scenarios/multigen_starv.json", "multigen_starv",
        n_gen=2, dur_per_gen=2520, dt=10)

print("\nDone!")
