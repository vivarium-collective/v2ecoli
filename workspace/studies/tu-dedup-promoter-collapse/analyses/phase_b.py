"""Phase B: two static reads against the refitted cache (out/cache_tudedup).

Axis 2  restored-dnag-tus-escape-the-constraint
        Z@R with all TFs bound, per dnaG-bearing TU. The fit's probability-
        validity constraint is 0 <= Z @ R <= 1 (promoter_fitting.py:489); the
        all-TFs-bound row of Z selects alpha_i + sum_tf r_{i,tf}. A TU pinned
        AT the boundary has that sum ~ 0.

Axis 3  lexa-still-regulates-rpsup3
        Where LexA's delta_prob edge sits among the three dnaG-bearing TUs.
"""
from __future__ import annotations
import json, sys
import dill, numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "out/cache_tudedup"
TU_IDS = ["TU00352", "TU00434", "TU00435"]
LEXA = "PC00010"          # LexA dimer, the regulator carried at this locus
EPS = 1e-9                # the band's "off the boundary" threshold

sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
ti = sd["configs"]["ecoli-transcript-initiation"]
ids = [str(x) for x in ti["rna_data"]["id"]]
basal = np.asarray(ti["basal_prob"], dtype=float)
D = np.asarray(ti["delta_prob_matrix"], dtype=float)

# tf_ids: column order of delta_prob_matrix
tfb = sd["configs"]["ecoli-tf-binding"]
tf_ids = [str(x) for x in (tfb.get("tf_ids") or tfb.get("tfs") or [])]

print(f"cache      : {CACHE}")
print(f"n TUs      : {len(ids)}   delta_prob_matrix {D.shape}   n TFs {len(tf_ids)}")

def find(tu):
    hits = [i for i, r in enumerate(ids) if r.startswith(tu)]
    return hits[0] if hits else None

rows = {}
for tu in TU_IDS:
    i = find(tu)
    rows[tu] = i
    if i is None:
        print(f"{tu}: ABSENT from this cache")
missing = [t for t, i in rows.items() if i is None]

# ---- Axis 2 -------------------------------------------------------------
print("\n=== axis 2: Z@R with all TFs bound ===")
n_off = 0
axis2 = {}
for tu, i in rows.items():
    if i is None:
        axis2[tu] = None
        continue
    deltas = D[i, :]
    nz = np.nonzero(deltas)[0]
    zr = float(basal[i] + deltas.sum())
    off = zr > EPS
    n_off += bool(off)
    axis2[tu] = dict(row=i, rna_id=ids[i], basal=float(basal[i]),
                     sum_delta=float(deltas.sum()), ZR_all_bound=zr,
                     off_boundary=bool(off),
                     tfs={(tf_ids[j] if j < len(tf_ids) else str(j)): float(deltas[j]) for j in nz})
    print(f"{tu} ({ids[i]}): basal={basal[i]:.6e}  sum_delta={deltas.sum():+.6e} "
          f" Z@R={zr:.6e}  -> {'OFF boundary' if off else 'PINNED at boundary'}")
    for j in nz:
        name = tf_ids[j] if j < len(tf_ids) else str(j)
        print(f"      TF {name}: delta={deltas[j]:+.6e}")
print(f"\naxis 2 measured_value = {n_off}  (band [2, 3])")

# ---- Axis 3 -------------------------------------------------------------
print("\n=== axis 3: where LexA's edge sits ===")
lex_col = tf_ids.index(LEXA) if LEXA in tf_ids else None
axis3 = {}
if lex_col is None:
    print(f"{LEXA} not in tf_ids -- cannot grade")
    frac = None
else:
    edges = {tu: (float(D[i, lex_col]) if i is not None else None) for tu, i in rows.items()}
    nonzero = {tu: v for tu, v in edges.items() if v not in (None, 0.0)}
    for tu, v in edges.items():
        print(f"{tu}: LexA delta = {v}")
    on_rpsup3 = sum(1 for tu in ("TU00435",) if edges.get(tu))
    frac = (on_rpsup3 / len(nonzero)) if nonzero else 0.0
    axis3 = dict(edges=edges, n_edges=len(nonzero), on_TU00435=on_rpsup3, fraction=frac)
    print(f"\naxis 3 measured_value = {frac}  (band [0.99, 1.0])")

# ---- why: the cistron->TU fit is degenerate for identical cistron sets -----
axis_why = {}
STATE = None
for cand in (f"{CACHE.replace('cache_','parca_')}/parca_state.pkl",):
    import os
    if os.path.exists(cand):
        STATE = cand
if STATE:
    import pickle, scipy.sparse as sp
    st = pickle.load(open(STATE, "rb"))
    tr = st["process"]["transcription"]
    get = (lambda o, k: o[k]) if isinstance(tr, dict) else getattr
    exp = np.asarray(st["cell_specs"]["basal"]["expression"], float).ravel()
    M = sp.csc_matrix(get(tr, "cistron_tu_mapping_matrix"))
    cid = [str(x) for x in get(tr, "cistron_data")["id"]]
    rel = st["relation"]
    r2t = rel["rna_id_to_regulating_tfs"] if isinstance(rel, dict) \
        else rel.rna_id_to_regulating_tfs
    print(f"\n=== why: expression allocation ({STATE}) ===")
    for tu, i in rows.items():
        if i is None:
            continue
        cistrons = sorted(cid[r] for r in M[:, i].nonzero()[0])
        print(f"{tu}: expression={exp[i]:.6e}  regulating_tfs={r2t.get(ids[i])}  cistrons={cistrons}")
    groups = {}
    for j in range(M.shape[1]):
        groups.setdefault(tuple(sorted(M[:, j].nonzero()[0])), []).append(j)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    winner_takes_all = sum(1 for v in dup.values() if sum(exp[j] > 0 for j in v) == 1)
    print(f"\nTU groups with an identical cistron set: {len(dup)} "
          f"(covering {sum(len(v) for v in dup.values())} TUs)")
    print(f"  groups where ONE member holds all the expression: {winner_takes_all}/{len(dup)}")
    axis_why = dict(state=STATE,
                    expression={tu: (float(exp[i]) if i is not None else None)
                                for tu, i in rows.items()},
                    regulating_tfs={tu: (list(r2t.get(ids[i], [])) if i is not None else None)
                                    for tu, i in rows.items()},
                    identical_cistron_set_groups=len(dup),
                    winner_takes_all_groups=winner_takes_all)

out = dict(cache=CACHE, n_tus=len(ids), missing=missing, why=axis_why,
           axis2=dict(per_tu=axis2, n_off_boundary=n_off, band=[2, 3]),
           axis3=axis3)
with open("workspace/studies/tu-dedup-promoter-collapse/analyses/phase_b_outcomes.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote analyses/phase_b_outcomes.json")
