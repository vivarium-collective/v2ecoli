"""Grade the three simulated axes of promoter-specific-regulation.

Reuses v2ecoli.library.generational_decay unchanged (req-3) so this arm is
measured exactly as dnag-locus-targeted-fix and dnag-promoter-availability were,
and the 0.455 ceiling those studies established is directly comparable.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
from v2ecoli.library import generational_decay as gd

TREE = sys.argv[1]
ARMS = {"promoter-specific": "out/cache_promspec",
        "attribution-control": "out/cache_tudedup_full"}
BASELINE_DEFICIT = 47.4          # pc00010-repression-arithmetic's measured starting deficit
SEEDS = (0, 1, 2)

def gens_completed(arm, seed):
    p = (f"out/promoter-specific-regulation__{arm}__seed{seed}/"
         f"promoter-specific-regulation__{arm}__seed{seed}_summary.json")
    return json.load(open(p))

# ---- axis: lineage-completes ------------------------------------------------
print("=== lineage-completes  (band [8, 12], measure = min generations across seeds)")
lin = {}
for arm in ARMS:
    per_seed = []
    for s in SEEDS:
        d = gens_completed(arm, s)
        divided = [g for g in d["gens"] if g.get("divided")]
        per_seed.append(len(divided))
    lin[arm] = per_seed
    print(f"  {arm:20s} divided-generation counts {per_seed}   min={min(per_seed)}")
measured_lineage = min(lin["promoter-specific"])
print(f"  measured_value (promoter-specific) = {measured_lineage}")

# ---- axis: rescue-is-not-a-global-growth-artefact ---------------------------
print("\n=== rescue-is-not-a-global-growth-artefact  (band [0.67, 1.5], tau ratio gens 1-3)")
def med_tau(arm, upto=3):
    vals = []
    for s in SEEDS:
        d = gens_completed(arm, s)
        vals += [g["duration_min"] for g in d["gens"][:upto]
                 if g.get("divided") and g.get("duration_min")]
    return float(np.median(vals)), vals
t_test, v_t = med_tau("promoter-specific")
t_ctrl, v_c = med_tau("attribution-control")
print(f"  promoter-specific   median tau gens1-3 = {t_test:.1f} min   n={len(v_t)}")
print(f"  attribution-control median tau gens1-3 = {t_ctrl:.1f} min   n={len(v_c)}")
ratio = t_test / t_ctrl
print(f"  measured_value = {ratio:.4f}")

# ---- axis: dnag-transcription-recovers --------------------------------------
print("\n=== dnag-transcription-recovers  (band [0.5, 1.0], fraction of log deficit closed)")
print(f"  (dnag-locus-targeted-fix measured the achievable CEILING at 0.455)")
out = {}
for arm, cache in ARMS.items():
    # compare at generation 3: the last generation every control seed reaches,
    # so both arms are measured on the same generations rather than on the
    # corrected arm's longer lineage.
    pg = gd.per_generation(f"{TREE}/{arm}", seeds=SEEDS, generations=(3,), cache_dir=cache)
    defs = []
    for s, gens in pg.items():
        for g, v in gens.items():
            if v.get("dnag", 0) > 0:
                defs.append(v["peer_median"] / v["dnag"])
    if defs:
        d = float(np.median(defs))
        closed = 1 - math.log(d) / math.log(BASELINE_DEFICIT)
        out[arm] = (d, closed)
        print(f"  {arm:20s} deficit={d:9.2f}x   fraction closed={closed:.4f}   n_seeds={len(defs)}")
    else:
        print(f"  {arm:20s} no dnaG signal recovered")

json.dump({"lineage": lin, "tau_ratio": ratio,
           "tau_medians": {"test": t_test, "control": t_ctrl},
           "deficit": {k: {"deficit": v[0], "closed": v[1]} for k, v in out.items()}},
          open("workspace/studies/promoter-specific-regulation/analyses/phase_c_outcomes.json", "w"),
          indent=2)
print("\nwrote analyses/phase_c_outcomes.json")
