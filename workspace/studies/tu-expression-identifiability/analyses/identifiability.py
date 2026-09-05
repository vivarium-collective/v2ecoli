"""Axes 1-4 of tu-expression-identifiability: static linear algebra, no simulation.

ParCa fits TU expression from cistron abundances by nonnegative least squares
(transcription.py:1310 -> fast_nnls). fast_nnls decomposes the problem into
connected components and solves each independently, so identifiability is a
per-block property. Where a block's matrix has rank below its column count the
optimum is a FACE, not a point, and NNLS returns an arbitrary vertex of it.

The alternative optimum used throughout is the MINIMUM-NORM point of the same
face -- deterministic, and it spreads mass across degenerate columns instead of
concentrating it. It is a PROBE for whether the choice matters, never a proposed
remedy: an even split has no more biological warrant than a vertex.

    min ||x||   s.t.  M x = M x*,  x >= 0

solved as a ridge-regularised NNLS on the augmented system [M; sqrt(eps) I],
which converges to the minimum-norm optimum as eps -> 0.

Run:  .venv/bin/python3 <this> [parca_state.pkl]
"""
from __future__ import annotations
import json, pickle, sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.sparse.csgraph import connected_components

STATE = sys.argv[1] if len(sys.argv) > 1 else "models/parca/parca_state.pkl.gz"
EPS = 1e-10          # ridge weight; small enough that the fit is unchanged to ~1e-12
RANK_TOL_FACTOR = 10  # blocks within this factor of the SVD tolerance are reported separately

opener = (lambda p: __import__("gzip").open(p, "rb")) if STATE.endswith(".gz") else (lambda p: open(p, "rb"))
st = pickle.load(opener(STATE))
tr = st["process"]["transcription"]
M = sp.csc_matrix(tr.cistron_tu_mapping_matrix).astype(float)
rna_ids = [str(x) for x in tr.rna_data["id"]]
exp = np.asarray(st["cell_specs"]["basal"]["expression"], float).ravel()
nC, nT = M.shape
print(f"state: {STATE}")
print(f"cistron_tu_mapping_matrix: {nC} cistrons x {nT} TUs\n")

# --- fast_nnls' own block decomposition -------------------------------------
big = sp.bmat([[None, M], [M.T, None]]).tocsr()
_, lab = connected_components(big, directed=False)
blocks: dict[int, tuple[list, list]] = {}
for j in range(nT):
    blocks.setdefault(lab[nC + j], ([], []))[1].append(j)
for i in range(nC):
    blocks.setdefault(lab[i], ([], []))[0].append(i)

under, marginal = [], []
tot_nullity = 0
for k, (rows, cols) in blocks.items():
    if len(cols) < 2:
        continue
    sub = M[rows][:, cols].toarray() if rows else np.zeros((0, len(cols)))
    if sub.size == 0:
        continue
    sv = np.linalg.svd(sub, compute_uv=False)
    tol = max(sub.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    rank = int((sv > tol).sum())
    if rank < len(cols):
        under.append((k, rows, cols, rank))
        tot_nullity += len(cols) - rank
        if sv.size and sv[sv > tol].min() < tol * RANK_TOL_FACTOR:
            marginal.append(k)

und_tus = sorted({c for _, _, cols, _ in under for c in cols})
print("=== axis 1: underdetermined-tu-count  (pin, <= 460)")
print(f"  rank-deficient blocks : {len(under)}")
print(f"  TUs in them           : {len(und_tus)}  ({len(und_tus)/nT:.1%} of {nT})")
print(f"  total nullity         : {tot_nullity}   (free parameters the cistron data never constrains)")
print(f"  marginal-rank blocks  : {len(marginal)}   (req-4: within {RANK_TOL_FACTOR}x of the SVD tolerance)")

# --- minimum-norm optimum, per underdetermined block ------------------------
alt = exp.copy()
for _, rows, cols, _ in under:
    sub = M[rows][:, cols].toarray()
    target = sub @ exp[cols]                       # the projection every optimum shares
    aug = np.vstack([sub, np.sqrt(EPS) * np.eye(len(cols))])
    rhs = np.concatenate([target, np.zeros(len(cols))])
    x, _ = nnls(aug, rhs)
    alt[cols] = x

# --- axis 2: is the alternative genuinely equally optimal? ------------------
cexp = np.asarray(st["cell_specs"]["basal"]["fit_cistron_expression"], float).ravel()
def resid(v):
    r = M.dot(v)
    return float(np.linalg.norm(r / r.sum() - cexp / cexp.sum()))
r0, r1 = resid(exp), resid(alt)
print(f"\n=== axis 2: cistron-fit-is-indifferent-to-the-split  (<= 1e-9)")
print(f"  residual, NNLS vertex   : {r0:.6e}")
print(f"  residual, minimum-norm  : {r1:.6e}")
print(f"  measured_value (change) : {abs(r1 - r0):.3e}")

# --- axis 3: are the vertex zeros liftable? ---------------------------------
# EXACT test, not the min-norm witness. A zero at coordinate j is liftable iff
# some direction d stays on the optimal face (sub @ d = 0), raises j, and does
# not drive any other currently-zero coordinate negative:
#
#   find d   s.t.  sub @ d = 0,  d_j >= 1,  d_i >= 0 for zero coords i != j
#
# Using the min-norm point as the witness instead only ever UNDERCOUNTS: it
# finds one particular alternative, not every one.
from scipy.optimize import linprog

def liftable(sub, xstar, local_j):
    n = sub.shape[1]
    zero_idx = [i for i in range(n) if xstar[i] == 0.0]
    bounds = []
    for i in range(n):
        if i == local_j:
            bounds.append((1.0, None))          # strictly lift j
        elif i in zero_idx:
            bounds.append((0.0, None))          # cannot go negative
        else:
            bounds.append((None, None))         # interior coord: either sign
    res = linprog(c=np.zeros(n), A_eq=sub, b_eq=np.zeros(sub.shape[0]),
                  bounds=bounds, method="highs")
    return bool(res.success)

zeros, lifted = [], []
for _, rows, cols, _ in under:
    xstar = exp[cols]
    if xstar.sum() <= 0:
        continue
    sub = M[rows][:, cols].toarray()
    for local_j, j in enumerate(cols):
        if exp[j] != 0.0:
            continue
        zeros.append(j)
        if liftable(sub, xstar, local_j):
            lifted.append(j)
print(f"\n=== axis 3: vertex-zeros-admit-equal-cost-alternatives  (band [0.8, 1.0])")
print(f"  vertex zeros in underdetermined blocks with nonzero block total: {len(zeros)}")
print(f"  demonstrably liftable at equal cost                            : {len(lifted)}")
a3 = len(lifted) / len(zeros) if zeros else float("nan")
print(f"  measured_value = {a3:.4f}   (exact: LP feasibility over the whole optimal face)")

# --- axis 4: does the choice move much? -------------------------------------
moved = 0
for j in und_tus:
    a, b = exp[j], alt[j]
    if a == 0 and b == 0:
        continue
    if a == 0 or b == 0 or max(a / b, b / a) > 2:
        moved += 1
a4 = moved / len(und_tus) if und_tus else float("nan")
print(f"\n=== axis 4: alternative-optimum-moves-many-tus  (band [0.3, 1.0])")
print(f"  underdetermined TUs moving >2x : {moved} of {len(und_tus)}")
print(f"  measured_value = {a4:.4f}")

json.dump({"state": STATE, "n_tus": nT,
           "axis1_underdetermined_tus": len(und_tus), "axis1_blocks": len(under),
           "axis1_nullity": tot_nullity, "axis1_marginal_blocks": len(marginal),
           "axis2_residual_change": abs(r1 - r0),
           "axis3_zeros": len(zeros), "axis3_lifted": len(lifted), "axis3_value": a3,
           "axis4_moved": moved, "axis4_value": a4},
          open("workspace/studies/tu-expression-identifiability/analyses/static_outcomes.json", "w"),
          indent=2)
print("\nwrote analyses/static_outcomes.json")
