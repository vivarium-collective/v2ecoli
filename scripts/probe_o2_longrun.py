"""Issue #143 — longer seed-0 baseline run: does O2/CO2 exchange climb toward the
report-card steady value (~-0.48 O2, +1.7 CO2) as the cell grows and its metabolite
pools dilute below their homeostatic targets? Logs every LOG_EVERY ticks.

Run:  PYTHONPATH=.deps:. python scripts/probe_o2_longrun.py  > out/o2_longrun.log 2>&1
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from v2ecoli import build_composite

N_TICKS = 4000
LOG_EVERY = 20
c = build_composite("ecoli_baseline", seed=0, cache_dir="out/cache")

# external index map (fixed order from metabolism.externalMoleculeIDs)
def find_metabolism(node):
    if isinstance(node, dict):
        for v in node.values():
            r = find_metabolism(v)
            if r is not None:
                return r
    else:
        if hasattr(node, "externalMoleculeIDs") and hasattr(node, "model"):
            return node
    return None

metab = find_metabolism(c.state)
ext_ids = list(metab.externalMoleculeIDs)
def idx_of(name):
    return ext_ids.index(name) if name in ext_ids else None
IDX = {"glucose": idx_of("GLC[p]"), "O2": idx_of("OXYGEN-MOLECULE[p]"),
       "CO2": idx_of("CARBON-DIOXIDE[p]"), "ammonium": idx_of("AMMONIUM[c]"),
       "acetate": idx_of("ACET[p]")}

def read_first_agent(state):
    agents = state.get("agents") or {}
    if not agents:
        return None, None
    aid = "0" if "0" in agents else sorted(agents)[0]
    ag = agents[aid]
    return aid, ag

print("tick   agent  gen   drymass_fg | glucose     O2       CO2     NH4    | O2:glc  RQ")
for t in range(1, N_TICKS + 1):
    c.run(1)
    if t % LOG_EVERY:
        continue
    aid, ag = read_first_agent(c.state)
    if ag is None:
        print(f"{t}: no agents (extinct?)"); break
    fr = ((ag.get("listeners") or {}).get("fba_results") or {})
    mass = ((ag.get("listeners") or {}).get("mass") or {})
    dm = mass.get("dry_mass")
    gen = ag.get("global", {}).get("generation") if isinstance(ag.get("global"), dict) else None
    eef = fr.get("external_exchange_fluxes")
    if eef is None or len(eef) == 0:
        continue
    eef = np.asarray(eef, float)
    v = {k: (eef[i] if i is not None else float("nan")) for k, i in IDX.items()}
    glc, o2, co2 = v["glucose"], v["O2"], v["CO2"]
    o2glc = abs(o2/glc) if glc else float("nan")
    rq = abs(co2/o2) if o2 else float("nan")
    try:
        dm_s = f"{float(dm):8.2f}"
    except Exception:
        dm_s = f"{str(dm)[:8]:>8}"
    print(f"{t:5d} {str(aid):>6} {str(gen):>4} {dm_s} | {glc:8.4f} {o2:8.4f} "
          f"{co2:8.4f} {v['ammonium']:7.3f} | {o2glc:6.3f} {rq:6.3f}", flush=True)
print("DONE")
