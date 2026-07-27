"""Issue #143 probe — read O2/CO2/glucose exchange directly from the baseline
FBA solution and compare against the external_exchange_fluxes listener, plus the
O2-consuming reaction fluxes (CYTBO etc.), on a fresh short seed-0 run.

Goal: discriminate (i) real low respiration vs (ii) instrumentation under-report.
Run:  PYTHONPATH=.deps:. .venv-python scripts/probe_o2_fba_vs_listener.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from v2ecoli import build_composite

N_TICKS = 8
c = build_composite("ecoli_baseline", seed=0, cache_dir="out/cache")

# --- locate the metabolism process instance in the composite tree ---
def find_metabolism(node, path=()):
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += find_metabolism(v, path + (k,))
    else:
        cls = type(node).__name__
        if cls == "Metabolism" or (hasattr(node, "externalMoleculeIDs")
                                   and hasattr(node, "model")):
            found.append((path, node))
    return found

metab = None
for state in (getattr(c, "state", None), getattr(c, "composition", None)):
    if state is None:
        continue
    hits = find_metabolism(state)
    if hits:
        metab = hits[0][1]
        print("metabolism process at:", hits[0][0])
        break

if metab is None:
    # fallback: dig through composite internal process registry
    for attr in dir(c):
        try:
            v = getattr(c, attr)
        except Exception:
            continue
        hits = find_metabolism(v) if isinstance(v, dict) else []
        if hits:
            metab = hits[0][1]
            print("metabolism via c.%s at %s" % (attr, hits[0][0]))
            break

assert metab is not None, "could not locate Metabolism process"

ext_ids = list(metab.externalMoleculeIDs)
fba = metab.model.fba
rxn_ids = list(fba.getReactionIDs())
print("n external molecules:", len(ext_ids), " n reactions:", len(rxn_ids))

def idx_of(ids, name):
    for i, x in enumerate(ids):
        if x == name:
            return i
    return None

TARGETS = {
    "glucose": "GLC[p]",
    "O2": "OXYGEN-MOLECULE[p]",
    "CO2": "CARBON-DIOXIDE[p]",
    "ammonium": "AMMONIUM[c]",
    "acetate": "ACET[p]",
}
ext_idx = {k: idx_of(ext_ids, v) for k, v in TARGETS.items()}
print("external indices:", ext_idx)

# Which reactions touch O2 / CO2 (from the FBA stoichiometry)?
stoich = fba.reactionStoich
def rxns_touching(mol):
    out = []
    for rid, st in stoich.items():
        for m, coeff in st.items():
            if m == mol or m == mol + "[c]" or m == mol + "[p]" or m.startswith(mol + "["):
                out.append((rid, coeff))
    return out
o2_rxns = rxns_touching("OXYGEN-MOLECULE")
co2_rxns = rxns_touching("CARBON-DIOXIDE")
print("reactions touching OXYGEN-MOLECULE: %d ; CARBON-DIOXIDE: %d" % (len(o2_rxns), len(co2_rxns)))
rxn_index = {r: i for i, r in enumerate(rxn_ids)}

# does the media make O2 available?
def read_env(state):
    agents = (state.get("agents") or {})
    ag = agents.get("0") or (next(iter(agents.values())) if agents else {})
    env = ag.get("environment") or {}
    return env

# --- run and read listener each tick ---
def read_agent_listener(state):
    agents = (state.get("agents") or {})
    ag = agents.get("0") or (next(iter(agents.values())) if agents else {})
    fr = ((ag.get("listeners") or {}).get("fba_results") or {})
    return fr

# check media exchange_data on the first environment we can read
c.run(1)
env0 = read_env(c.state)
xd = (env0.get("exchange_data") or {})
uncon = set(xd.get("unconstrained") or [])
con = dict(xd.get("constrained") or {})
print("\nmedia_id:", env0.get("media_id"))
print("O2 in unconstrained?", "OXYGEN-MOLECULE[p]" in uncon,
      " O2 in constrained?", "OXYGEN-MOLECULE[p]" in con)
print("CO2 in unconstrained?", "CARBON-DIOXIDE[p]" in uncon,
      " CO2 in constrained?", "CARBON-DIOXIDE[p]" in con)
print("n unconstrained:", len(uncon), " sample:", sorted(uncon)[:8])
print("n constrained:", len(con))

print("\ntick |   glucose      O2        CO2      ammonium   acetate   |  O2:glc  RQ | O2_by_rxns")
rows = []
for t in range(N_TICKS):
    if t > 0:
        c.run(1)
    fr = read_agent_listener(c.state)
    eef = fr.get("external_exchange_fluxes")
    rf = fr.get("reaction_fluxes")
    if eef is None or len(eef) == 0:
        print(f"{t:4d} | (no exchange emitted yet)")
        continue
    eef = np.asarray(eef, float)
    vals = {k: (eef[i] if i is not None else float("nan")) for k, i in ext_idx.items()}
    glc, o2, co2 = vals["glucose"], vals["O2"], vals["CO2"]
    o2glc = abs(o2 / glc) if glc else float("nan")
    rq = abs(co2 / o2) if o2 else float("nan")
    # net O2 consumed by internal reactions (native reaction-flux units)
    o2net = float("nan")
    if rf is not None and len(rf) == len(rxn_ids):
        rf = np.asarray(rf, float)
        o2net = sum(coeff * rf[rxn_index[rid]] for rid, coeff in o2_rxns
                    if rid in rxn_index)
    print(f"{t:4d} | {glc:9.4f} {o2:9.4f} {co2:9.4f} {vals['ammonium']:9.4f} "
          f"{vals['acetate']:9.4f} | {o2glc:6.3f} {rq:6.3f} | {o2net:9.4f}")
    rows.append(vals)

if rows:
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in TARGETS}
    print("\nMEAN over ticks (mmol/gDCW/h, listener/report-card units):")
    for k in TARGETS:
        print(f"  {k:9s}: {mean[k]:+.4f}")
    print(f"  O2:glucose ratio = {abs(mean['O2']/mean['glucose']):.4f}  (v1~0.14, phys~1.5-2)")
    print(f"  RQ = CO2/O2      = {abs(mean['CO2']/mean['O2']):.4f}  (respiratory~1)")
