"""Phase 5 PC algorithm v2 — proper conditional-independence skeleton.

Improves the v1 correlation-only pilot (55.6% recall, 13.9% precision)
by implementing the proper PC algorithm with partial-correlation
conditional independence tests:

  Phase 1 (skeleton):
    Start with complete undirected graph.
    For increasing conditioning-set size |S| = 0, 1, 2, ...
      For each edge X-Y:
        For each S ⊆ neighbors(X) \\ {Y} of size |S|:
          If X ⊥ Y | S (partial correlation test), remove edge,
          record sepset(X, Y) = S.

  Phase 2 (orient v-structures):
    For each unshielded triple X-Z-Y (X-Z and Z-Y edges, but X-Y NOT
    adjacent): if Z is NOT in sepset(X, Y), orient X → Z ← Y.

Run from worktree root:
    python scripts/run_causal_pc_v2.py [--alpha 0.05] [--max-cond-size 2]
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import basico

OUT_ROOT = Path(".pbg/runs/causal-pc-v2")
FIG_DIR = Path("reports/figures/pdmp-05")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})

MODEL_PATH = "v2ecoli/models/sbml/millard2017_central_metabolism.xml"
NODES = ["G6P", "F6P", "FDP", "GAP", "PEP", "PYR", "ACCOA", "CIT", "AKG"]


def _to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _save_html(name: str, title: str, caption: str, pinned_h: int = 900):
    b64 = _to_b64()
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{pinned_h}px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{pinned_h}px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
p.caption{{color:#475569;font-size:0.85em;line-height:1.4}}
.fig{{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.fig img{{max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain}}
.tag{{display:inline-block;background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p><span class="tag">real-data</span><span class="tag">PC v2 (proper CI)</span></p>
<div class="fig"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
<p class="caption">{caption}</p>
</div></body></html>"""
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def run_millard(perturb_init: dict | None = None,
                duration: float = 500.0,
                knockout_param: tuple[str, float] | None = None) -> dict:
    """Capture state at t=duration (default 500s = TRANSIENT, not steady-state).

    At t=5000s every replicate converges to the same steady state regardless of
    IC perturbations (Millard is deterministic + has a unique attractor), so
    metabolites are perfectly correlated and PC can't extract any structure.
    At transient timepoints the trajectories differ enough to recover causal
    structure.
    """
    basico.load_model(MODEL_PATH)
    if perturb_init:
        for sp, val in perturb_init.items():
            try: basico.set_species(sp, initial_concentration=float(val))
            except Exception: pass
    if knockout_param:
        try: basico.set_reaction_parameters(name=knockout_param[0], value=knockout_param[1])
        except Exception: pass
    try:
        ts = basico.run_time_course(start_time=0, duration=duration, intervals=20,
                                    use_sbml_id=True, update_model=True)
        last = ts.iloc[-1]
        return {n: float(last.get(n, np.nan)) for n in NODES}
    except Exception:
        return {n: np.nan for n in NODES}


def ground_truth_edges() -> set[tuple[str, str]]:
    basico.load_model(MODEL_PATH)
    rxns = basico.get_reactions()
    edges: set[tuple[str, str]] = set()
    for _, row in rxns.iterrows():
        scheme = str(row.get("scheme", ""))
        sep = "=" if "=" in scheme else ("->" if "->" in scheme else None)
        if not sep: continue
        lhs, rhs = scheme.split(sep, 1)
        substrates = {s.strip() for s in lhs.split(";")[0].replace("+", " ").split() if s.strip()}
        products = {s.strip() for s in rhs.split(";")[0].replace("+", " ").split() if s.strip()}
        for s in substrates:
            for p in products:
                if s in NODES and p in NODES and s != p:
                    edges.add((s, p))
    return edges


def partial_corr_test(X: np.ndarray, i: int, j: int, condset: list[int],
                       alpha: float = 0.05) -> tuple[bool, float, float]:
    """Test if column i ⊥ column j of X given columns in condset.

    Uses partial correlation via residualization + Fisher z-transform p-value.
    Returns (is_independent, p_value, partial_r).
    """
    n = X.shape[0]
    if condset:
        Z = X[:, condset]
        Z_int = np.column_stack([np.ones(n), Z])
        try:
            beta_i, *_ = np.linalg.lstsq(Z_int, X[:, i], rcond=None)
            beta_j, *_ = np.linalg.lstsq(Z_int, X[:, j], rcond=None)
        except np.linalg.LinAlgError:
            return False, 1.0, 0.0
        res_i = X[:, i] - Z_int @ beta_i
        res_j = X[:, j] - Z_int @ beta_j
    else:
        res_i = X[:, i] - X[:, i].mean()
        res_j = X[:, j] - X[:, j].mean()
    si = res_i.std(); sj = res_j.std()
    if si < 1e-12 or sj < 1e-12:
        return True, 1.0, 0.0
    r = float(np.mean(res_i * res_j) / (si * sj))
    r = max(-0.9999, min(0.9999, r))
    k = len(condset)
    if n - k - 3 <= 0:
        return False, 1.0, r
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - k - 3)
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(z) / se)))
    return p > alpha, p, r


def pc_skeleton(X: np.ndarray, alpha: float = 0.05,
                max_cond_size: int = 2) -> tuple[set[frozenset], dict[frozenset, list[int]]]:
    n_vars = X.shape[1]
    edges: set[frozenset] = {frozenset({i, j}) for i in range(n_vars)
                              for j in range(i+1, n_vars)}
    sepset: dict[frozenset, list[int]] = {}

    for cond_size in range(max_cond_size + 1):
        edges_to_remove = []
        for edge in list(edges):
            i, j = sorted(edge)
            nbrs_i = [k for k in range(n_vars) if k != i and k != j
                      and frozenset({i, k}) in edges]
            if len(nbrs_i) < cond_size:
                continue
            for S in combinations(nbrs_i, cond_size):
                indep, p, r = partial_corr_test(X, i, j, list(S), alpha=alpha)
                if indep:
                    edges_to_remove.append(edge)
                    sepset[edge] = list(S)
                    break
        for e in edges_to_remove:
            edges.discard(e)
        print(f"   cond_size={cond_size}: removed {len(edges_to_remove)} edges, {len(edges)} remain")
    return edges, sepset


def orient_v_structures(edges: set[frozenset], sepset: dict[frozenset, list[int]],
                         n_vars: int) -> set[tuple[int, int]]:
    directed: set[tuple[int, int]] = set()
    for z in range(n_vars):
        nbrs = [k for k in range(n_vars) if frozenset({z, k}) in edges]
        for x, y in combinations(nbrs, 2):
            if frozenset({x, y}) in edges:
                continue
            sep = sepset.get(frozenset({x, y}))
            if sep is not None and z not in sep:
                directed.add((x, z))
                directed.add((y, z))
    return directed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-obs", type=int, default=80)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-cond-size", type=int, default=2)
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)

    print("1. Ground-truth edges from Millard...")
    gt = ground_truth_edges()
    print(f"   {len(gt)} substrate→product edges over {len(NODES)} nodes")

    print(f"\n2. Generating {args.n_obs} observational runs (TRANSIENT timepoints + larger IC perturbations)...")
    basico.load_model(MODEL_PATH)
    init_concs = {n: float(basico.get_species(n).iloc[0]["initial_concentration"]) for n in NODES}
    obs = []
    t0 = time.time()
    # Use a MIX of perturbation magnitudes + a MIX of capture times to create
    # multi-dimensional variation. Otherwise Millard's deterministic attractor
    # collapses all replicates to the same steady state with rank-1 correlations.
    capture_times = [200.0, 500.0, 1000.0]
    for i in range(args.n_obs):
        # Larger per-species independent perturbation (3x noise)
        perturb = {n: init_concs[n] * max(0.05, 1 + 0.3 * rng.standard_normal()) for n in NODES}
        t_capture = float(rng.choice(capture_times))
        out = run_millard(perturb_init=perturb, duration=t_capture)
        if any(np.isnan(v) for v in out.values()):
            continue
        obs.append(out)
        if (i + 1) % 20 == 0: print(f"   {i+1}/{args.n_obs} done ({time.time()-t0:.1f}s)")
    print(f"   {len(obs)} successful ({time.time()-t0:.1f}s wall)")

    X = np.array([[o[n] for n in NODES] for o in obs])

    print(f"\n3. PC algorithm skeleton (α={args.alpha}, max |S|={args.max_cond_size})...")
    edges_und, sepset = pc_skeleton(X, alpha=args.alpha, max_cond_size=args.max_cond_size)
    print(f"   skeleton has {len(edges_und)} undirected edges")

    print(f"\n4. Orienting v-structures...")
    directed = orient_v_structures(edges_und, sepset, len(NODES))
    directed_named = {(NODES[i], NODES[j]) for (i, j) in directed}
    print(f"   {len(directed_named)} edges oriented")

    gt_und = {frozenset({s, t}) for (s, t) in gt}
    tp_und = edges_und & gt_und
    recall_und = len(tp_und) / max(len(gt_und), 1)
    precision_und = len(tp_und) / max(len(edges_und), 1)
    tp_dir = directed_named & gt
    recall_dir = len(tp_dir) / max(len(gt), 1)
    precision_dir = len(tp_dir) / max(len(directed_named), 1) if directed_named else 0
    print(f"\n   UNDIRECTED skeleton recall:    {recall_und:.2%}  ({len(tp_und)}/{len(gt_und)})")
    print(f"   UNDIRECTED skeleton precision: {precision_und:.2%}  ({len(tp_und)}/{len(edges_und)})")
    print(f"   DIRECTED recall:               {recall_dir:.2%}  ({len(tp_dir)}/{len(gt)})")
    print(f"   DIRECTED precision:            {precision_dir:.2%}  ({len(tp_dir)}/{len(directed_named) if directed_named else 0})")

    v1_path = Path(".pbg/runs/causal-pc-pilot/recovered_dag.json")
    v1_result = json.loads(v1_path.read_text()) if v1_path.exists() else None

    result = {
        "n_obs": len(obs), "alpha": args.alpha, "max_cond_size": args.max_cond_size,
        "n_gt": len(gt),
        "skeleton_edges": [sorted(e) for e in edges_und],
        "directed_edges": [list(e) for e in directed_named],
        "undirected_recall": recall_und, "undirected_precision": precision_und,
        "directed_recall": recall_dir, "directed_precision": precision_dir,
    }
    (OUT_ROOT / "result.json").write_text(json.dumps(result, indent=2))

    # ---- Viz ----
    print("\nGenerating viz:")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = ["v1 correlation", "v2 PC (proper CI)"]
    if v1_result:
        v1_recall_und = v1_result.get("recall", 0.0)
        v1_precision_und = v1_result.get("precision", 0.0)
    else:
        v1_recall_und = v1_precision_und = 0.0
    width = 0.35
    x = np.arange(len(labels))
    recalls = [v1_recall_und, recall_und]
    precisions = [v1_precision_und, precision_und]
    ax.bar(x - width/2, recalls, width, label="Skeleton recall", color="#3b82f6", alpha=0.85)
    ax.bar(x + width/2, precisions, width, label="Skeleton precision", color="#ef4444", alpha=0.85)
    ax.axhline(0.80, color="green", ls="--", lw=1.5, label="80% test threshold")
    for i, (r, pp) in enumerate(zip(recalls, precisions)):
        ax.text(i - width/2, r + 0.02, f"{r:.0%}", ha="center", fontsize=10)
        ax.text(i + width/2, pp + 0.02, f"{pp:.0%}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title(f"PC algorithm: v1 correlation vs v2 proper CI (N_obs={len(obs)}, α={args.alpha})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    _save_html("pc_v2_vs_v1_real",
               "PC algorithm — v1 (correlation) vs v2 (proper CI)",
               (
                 f"Real comparison: v1 (correlation-only |r|>0.4) vs v2 (proper PC with "
                 f"partial-correlation conditional independence tests, α={args.alpha}, |S| up to "
                 f"{args.max_cond_size}). v2 should achieve higher precision via CI tests that "
                 f"rule out spurious edges. N_obs = {len(obs)} observational runs."
               ))

    # Heatmap
    n_nodes = len(NODES)
    adj_gt = np.zeros((n_nodes, n_nodes))
    adj_rec = np.zeros((n_nodes, n_nodes))
    for s, t in gt: adj_gt[NODES.index(s), NODES.index(t)] = 1
    for s, t in directed_named: adj_rec[NODES.index(s), NODES.index(t)] = 1
    combined = adj_gt + 2 * adj_rec
    fig, ax = plt.subplots(figsize=(10, 9))
    rgb = np.zeros((n_nodes, n_nodes, 3))
    color_map = {0: (1, 1, 1), 1: (0.99, 0.79, 0.79), 2: (0.99, 0.84, 0.66), 3: (0.52, 0.94, 0.58)}
    for v, c in color_map.items():
        mask = combined == v
        for k in range(3): rgb[..., k][mask] = c[k]
    ax.imshow(rgb)
    ax.set_xticks(range(n_nodes)); ax.set_yticks(range(n_nodes))
    ax.set_xticklabels(NODES, rotation=45, ha="right")
    ax.set_yticklabels(NODES)
    ax.set_xlabel("To (column)"); ax.set_ylabel("From (row)")
    ax.set_title(f"PC v2 — recovered DAG vs ground truth\ngreen=TP ({len(tp_dir)}), red=FN, orange=FP")
    _save_html("pc_v2_dag_vs_groundtruth_real",
               "PC v2 — recovered DAG vs ground truth",
               (
                 f"PC v2 recovered directed DAG vs Millard reaction-graph ground truth. "
                 f"Green = TP ({len(tp_dir)}/{len(gt)}, recall={recall_dir:.0%}), "
                 f"red = FN, orange = FP."
               ),
               pinned_h=1100)

    print("\nDone.")


if __name__ == "__main__":
    main()
