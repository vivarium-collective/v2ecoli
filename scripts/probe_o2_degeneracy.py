"""Issue #143 — degeneracy test. In the non-respiratory regime (late in a
generation), is the FBA marginally INDIFFERENT to O2 uptake? A near-zero reduced
cost (col_dual) on the O2 external-exchange column means O2/CO2 sit on the LP's
null space (alternate optima) -> the exchange fluxes are not robustly pinned by
the homeostatic objective, so their time-average and the v1<->v2 delta are
fragile, not a robust biological signal.

Run:  PYTHONPATH=.deps:. python scripts/probe_o2_degeneracy.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from v2ecoli import build_composite

WARMUP = 520   # ticks to reach the non-respiratory late-generation regime
c = build_composite("ecoli_baseline", seed=0, cache_dir="out/cache")

def find_metabolism(node):
    if isinstance(node, dict):
        for v in node.values():
            r = find_metabolism(v)
            if r is not None:
                return r
    elif hasattr(node, "externalMoleculeIDs") and hasattr(node, "model"):
        return node
    return None

metab = find_metabolism(c.state)
fba = metab.model.fba
ext_ids = list(metab.externalMoleculeIDs)
exch_ids = list(fba._externalExchangeIDs)   # "external exchange - <mol>"
solver = fba._solver
flows = solver._flows

def exch_flow_idx(mol):
    name = "external exchange - " + mol
    return flows.get(name)

targets = {"O2": "OXYGEN-MOLECULE[p]", "CO2": "CARBON-DIOXIDE[p]",
           "glucose": "GLC[p]", "ammonium": "AMMONIUM[c]"}

print(f"warming up {WARMUP} ticks to reach the non-respiratory regime...")
for t in range(WARMUP):
    c.run(1)

# read the solver caches from the most-recent solve
col_primals = np.asarray(solver._col_primals, float)
col_duals = np.asarray(solver._col_duals, float)

print("\n(regime check) exchange primals & reduced costs at tick", WARMUP)
print(f"{'metabolite':10s} {'primal(flow)':>14s} {'reduced_cost':>14s}")
for k, mol in targets.items():
    fi = exch_flow_idx(mol)
    if fi is None:
        print(f"{k:10s}  (no exchange flow)"); continue
    print(f"{k:10s} {col_primals[fi]:14.6f} {col_duals[fi]:14.6e}")

# scale reference: reduced cost of the glucose exchange (an actively-constrained,
# strongly-determined uptake) vs O2 (hypothesised degenerate).
o2_rc = col_duals[exch_flow_idx("OXYGEN-MOLECULE[p]")]
glc_rc = col_duals[exch_flow_idx("GLC[p]")]
print("\nInterpretation:")
print(f"  |reduced_cost(O2)|  = {abs(o2_rc):.3e}")
print(f"  |reduced_cost(glc)| = {abs(glc_rc):.3e}")
print("  If |rc(O2)| << |rc(glc)| (near 0), the LP objective is marginally")
print("  indifferent to O2 -> respiratory exchange is degenerate (alternate optima).")

# Also report the min/max primal magnitude for O2-consuming reactions to show
# respiration is 'off' in this regime.
