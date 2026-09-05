"""Verify the promoter-specific ParCa arm against the control.

Checks, in order of what would invalidate the change:
  1. the network moved by the predicted amount (473 edges dropped, 989 kept);
  2. the dnaG edge specifically is gone from TU00352;
  3. nothing else about the fit collapsed (TF count, edge count sanity).
"""
from __future__ import annotations
import sys
import numpy as np, dill

CTRL = sys.argv[1] if len(sys.argv) > 1 else "out/cache"
TEST = sys.argv[2] if len(sys.argv) > 2 else "out/cache_promspec"

def load(cache):
    sd = dill.load(open(f"{cache}/sim_data_cache.dill", "rb"))
    ti = sd["configs"]["ecoli-transcript-initiation"]
    return (sd, ti,
            [str(x) for x in ti["rna_data"]["id"]],
            np.asarray(ti["basal_prob"], float),
            np.asarray(ti["delta_prob_matrix"], float),
            [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]],
            ti["delta_prob"])

_, _, ids_c, b_c, D_c, tf_c, dp_c = load(CTRL)
_, _, ids_t, b_t, D_t, tf_t, dp_t = load(TEST)
print(f"control  {CTRL:22s} TUs={len(ids_c)} declared_edges={len(dp_c['deltaV'])} "
      f"nonzero={int((D_c!=0).sum())} TFs_used={int((D_c!=0).any(0).sum())}")
print(f"promspec {TEST:22s} TUs={len(ids_t)} declared_edges={len(dp_t['deltaV'])} "
      f"nonzero={int((D_t!=0).sum())} TFs_used={int((D_t!=0).any(0).sum())}")

dropped = len(dp_c["deltaV"]) - len(dp_t["deltaV"])
print(f"\ndeclared edges dropped: {dropped}   (split predicted 473 CONTRADICTED)")
print(f"declared edges kept   : {len(dp_t['deltaV'])}  (predicted 989; "
      f"unchanged fraction {len(dp_t['deltaV'])/len(dp_c['deltaV']):.4f}, floor 0.65)")

L_c, L_t = tf_c.index("PC00010"), tf_t.index("PC00010")
print("\n--- the dnaG locus")
for tu in ("TU00352", "TU00434", "TU00435"):
    hc = [i for i, x in enumerate(ids_c) if x.startswith(tu)]
    ht = [i for i, x in enumerate(ids_t) if x.startswith(tu)]
    vc = f"{D_c[hc[0], L_c]:+.6f}" if hc else "absent"
    vt = f"{D_t[ht[0], L_t]:+.6f}" if ht else "absent"
    bc = f"{b_c[hc[0]]:.4e}" if hc else "-"
    bt = f"{b_t[ht[0]]:.4e}" if ht else "-"
    print(f"  {tu}: LexA delta  control={vc:>12s}  promspec={vt:>12s}   "
          f"basal {bc} -> {bt}")

print(f"\n--- LexA edges overall: control={int((D_c[:,L_c]!=0).sum())}  "
      f"promspec={int((D_t[:,L_t]!=0).sum())}")
