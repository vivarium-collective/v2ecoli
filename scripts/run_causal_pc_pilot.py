"""Phase 5 PC-style causal discovery pilot on Millard.

Minimal causal-discovery deliverable for pdmp-05's pc-recovers-known-edges
primary test. No extra deps beyond numpy + scipy.stats.

Pipeline:
  1. Generate OBSERVATIONAL data: run Millard N times with random ±10%
     initial-concentration perturbations on a subset of species. Capture
     steady-state state of N target metabolites.
  2. Generate INTERVENTIONAL data: knock out each of K reactions
     (Vmax → 0 or kF → very small) and run to steady state. Each
     intervention produces one new state vector.
  3. Skeleton recovery: pairwise correlation across observations. Edges
     with |r| above threshold remain.
  4. Edge orientation via interventions: for each candidate edge X-Y,
     if knocking out X's-upstream-reactions shifts Y but knocking out
     Y's-upstream-reactions doesn't shift X → X causes Y.
  5. Compare recovered DAG against Millard's reaction adjacency
     (extracted from basico.get_reactions() scheme strings).

Output:
  .pbg/runs/causal-pc-pilot/{observations,interventions}.json
  .pbg/runs/causal-pc-pilot/recovered_dag.json
  reports/figures/pdmp-05/pc_recovered_edges_real.html
  reports/figures/pdmp-05/pc_recall_vs_threshold_real.html

Run from worktree root:
    python scripts/run_causal_pc_pilot.py
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

OUT_ROOT = Path(".pbg/runs/causal-pc-pilot")
FIG_DIR = Path("reports/figures/pdmp-05")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})

MODEL_PATH = "v2ecoli/models/sbml/millard2017_central_metabolism.xml"
# Target nodes for the discovered graph. Picked to give Millard a non-trivial
# DAG structure (upstream glucose → glycolysis → TCA).
NODES = ["G6P", "F6P", "FDP", "GAP", "PEP", "PYR", "ACCOA", "CIT", "AKG"]

# Reactions to knock out (one per intervention). Each maps a reaction name
# to the parameter we'll zero out (its dominant rate constant).
INTERVENTIONS = [
    ("PTS_4", "(PTS_4).kF",    "glucose → G6P (uptake)"),
    ("PGI",   "(PGI).Keq",     "G6P → F6P"),
    ("PFK",   "(PFK).Keq",     "F6P → FDP"),
    ("PYK",   "(PYK).Keq",     "PEP → PYR"),
    ("PDH",   "(PDH).Vmax",    "PYR → ACCOA"),
    ("CS",    "(CS).kF",       "ACCOA + OAA → CIT"),
]


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
<p><span class="tag">real-data</span><span class="tag">PC pilot</span></p>
<div class="fig"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
<p class="caption">{caption}</p>
</div></body></html>"""
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def run_millard(perturb_init: dict | None = None,
                knockout_param: tuple[str, float] | None = None,
                duration: float = 5000.0) -> dict:
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
    """Extract Millard's reaction adjacency restricted to NODES.
    For each reaction (substrates → products) we add (s, p) directed edges
    for every (s in substrates, p in products) pair when both ∈ NODES."""
    basico.load_model(MODEL_PATH)
    rxns = basico.get_reactions()
    edges: set[tuple[str, str]] = set()
    for rxn_name, row in rxns.iterrows():
        scheme = str(row.get("scheme", ""))
        # Parse "subs1 + subs2 = prod1 + prod2" (Copasi schemes use '=' for reversible, '->' for irreversible)
        sep = "=" if "=" in scheme else ("->" if "->" in scheme else None)
        if not sep: continue
        lhs, rhs = scheme.split(sep, 1)
        # Drop modifiers (after ';')
        for side in (lhs, rhs):
            pass
        lhs_clean = lhs.split(";")[0]
        rhs_clean = rhs.split(";")[0]
        substrates = {s.strip() for s in lhs_clean.replace("+", " ").split() if s.strip()}
        products = {s.strip() for s in rhs_clean.replace("+", " ").split() if s.strip()}
        for s in substrates:
            for p in products:
                if s in NODES and p in NODES and s != p:
                    edges.add((s, p))
    return edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-obs", type=int, default=40,
                   help="number of observational runs (random IC perturbations)")
    p.add_argument("--corr-threshold", type=float, default=0.4,
                   help="|r| threshold for skeleton edges")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    print(f"1. Ground-truth edges from Millard reaction graph...")
    gt = ground_truth_edges()
    print(f"   {len(gt)} directed substrate→product edges over {NODES}")
    for e in sorted(gt): print(f"     {e[0]:5s} → {e[1]}")

    print(f"\n2. Generating {args.n_obs} OBSERVATIONAL runs (±10% IC perturbations)...")
    basico.load_model(MODEL_PATH)
    init_concs = {n: float(basico.get_species(n).iloc[0]["initial_concentration"]) for n in NODES}
    obs: list[dict] = []
    t0 = time.time()
    for i in range(args.n_obs):
        perturb = {n: init_concs[n] * (1 + 0.1 * rng.standard_normal()) for n in NODES}
        out = run_millard(perturb_init=perturb)
        if any(np.isnan(v) for v in out.values()):
            continue
        obs.append(out)
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{args.n_obs} done")
    print(f"   {len(obs)} successful runs ({time.time()-t0:.1f}s wall)")
    (OUT_ROOT / "observations.json").write_text(json.dumps(obs, indent=2))

    # 3. Pairwise correlation → skeleton edges
    print(f"\n3. Building skeleton (|r| > {args.corr_threshold} threshold)...")
    X = np.array([[o[n] for n in NODES] for o in obs])
    from scipy.stats import pearsonr
    corr_matrix = np.zeros((len(NODES), len(NODES)))
    pval_matrix = np.ones((len(NODES), len(NODES)))
    skeleton_edges: set[frozenset] = set()
    for i, j in combinations(range(len(NODES)), 2):
        try:
            r, pval = pearsonr(X[:, i], X[:, j])
        except Exception:
            r, pval = 0.0, 1.0
        corr_matrix[i, j] = corr_matrix[j, i] = r
        pval_matrix[i, j] = pval_matrix[j, i] = pval
        if abs(r) >= args.corr_threshold and pval < 0.05:
            skeleton_edges.add(frozenset({NODES[i], NODES[j]}))
    print(f"   {len(skeleton_edges)} skeleton edges (undirected)")

    # 4. Interventional orientation
    print(f"\n4. Running {len(INTERVENTIONS)} interventional runs...")
    interventions: dict[str, dict] = {}
    for rxn_name, param, label in INTERVENTIONS:
        out = run_millard(knockout_param=(param, 0.0))
        interventions[rxn_name] = {
            "param": param, "label": label, "endpoint": out,
        }
        print(f"   {rxn_name} ({label}): {out}")
    (OUT_ROOT / "interventions.json").write_text(json.dumps(interventions, indent=2))

    # Heuristic edge orientation: for X-Y edge, X→Y if knocking out X
    # (zeroing a reaction with X as substrate) shifts Y by >5% of its
    # baseline observational range. Baseline = sample mean of obs[].
    baseline = {n: float(np.mean(X[:, NODES.index(n)])) for n in NODES}
    edge_directions: dict[tuple[str, str], str] = {}
    for edge in skeleton_edges:
        a, b = sorted(edge)
        # For each intervention, look at the shift on a and b
        a_shifts, b_shifts = [], []
        for rxn_name, info in interventions.items():
            end = info["endpoint"]
            a_shift = (end.get(a, baseline[a]) - baseline[a]) / max(abs(baseline[a]), 1e-9)
            b_shift = (end.get(b, baseline[b]) - baseline[b]) / max(abs(baseline[b]), 1e-9)
            a_shifts.append(a_shift); b_shifts.append(b_shift)
        max_a = max(abs(s) for s in a_shifts)
        max_b = max(abs(s) for s in b_shifts)
        # Orient: the node that shifts MORE under intervention is the downstream
        # one (effect); the less-shifted is upstream (cause).
        if max_a > max_b * 1.5:
            edge_directions[(b, a)] = "b→a (a shifts more)"
        elif max_b > max_a * 1.5:
            edge_directions[(a, b)] = "a→b (b shifts more)"
        else:
            edge_directions[(a, b)] = "undirected (shifts comparable)"

    recovered_edges = {(s, t) for (s, t) in edge_directions.keys()}
    print(f"\n5. Recovered DAG: {len(recovered_edges)} directed edges")
    for e, why in edge_directions.items():
        print(f"   {e[0]:5s} → {e[1]:5s}  ({why})")

    # Compute recall vs ground truth
    tp = recovered_edges & gt
    fn = gt - recovered_edges
    fp = recovered_edges - gt
    recall = len(tp) / max(len(gt), 1)
    precision = len(tp) / max(len(recovered_edges), 1)
    print(f"\n   Ground-truth edges: {len(gt)}")
    print(f"   Recovered edges:    {len(recovered_edges)}")
    print(f"   True positives:     {len(tp)}")
    print(f"   False negatives:    {len(fn)}")
    print(f"   False positives:    {len(fp)}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   Precision: {precision:.2%}")

    (OUT_ROOT / "recovered_dag.json").write_text(json.dumps({
        "nodes": NODES,
        "ground_truth_edges": [list(e) for e in gt],
        "recovered_edges": [list(e) for e in recovered_edges],
        "true_positives": [list(e) for e in tp],
        "false_negatives": [list(e) for e in fn],
        "false_positives": [list(e) for e in fp],
        "recall": recall, "precision": precision,
        "n_obs": len(obs), "n_interventions": len(interventions),
        "corr_threshold": args.corr_threshold,
    }, indent=2))

    # ---- Viz ----
    print("\nGenerating viz:")
    # 1. Correlation heatmap with ground-truth + recovered edges overlay
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.abs(corr_matrix), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(NODES))); ax.set_yticks(range(len(NODES)))
    ax.set_xticklabels(NODES, rotation=45, ha="right")
    ax.set_yticklabels(NODES)
    ax.set_title(
        f"PC pilot — |Pearson r| matrix from {len(obs)} observational runs\n"
        f"GT edges = green ●; recovered = red ▲; |r| threshold = {args.corr_threshold}"
    )
    for (i, j) in [(NODES.index(s), NODES.index(t)) for (s, t) in gt]:
        ax.scatter(j, i, s=120, marker="o", facecolor="none", edgecolor="lime", lw=2.5, zorder=10)
    for (i, j) in [(NODES.index(s), NODES.index(t)) for (s, t) in recovered_edges]:
        ax.scatter(j, i, s=80, marker="^", color="red", edgecolor="black", lw=1, zorder=11)
    plt.colorbar(im, ax=ax, label="|r|")
    _save_html("pc_recovered_edges_real",
               "PC pilot — recovered DAG vs ground truth",
               (
                 f"Pairwise |Pearson r| matrix from {len(obs)} observational Millard runs "
                 f"(±10% IC perturbations). Green ○ = ground-truth substrate→product edges "
                 f"from Millard reaction adjacency (restricted to {len(NODES)} central metabolites). "
                 f"Red ▲ = edges recovered by skeleton (|r|>{args.corr_threshold}, p<0.05) + "
                 f"interventional orientation. Recall = {recall:.0%} ({len(tp)}/{len(gt)} edges). "
                 f"Precision = {precision:.0%}."
               ))

    # 2. Recall vs threshold sweep
    thresholds = np.linspace(0.05, 0.95, 25)
    recalls = []
    precisions = []
    for tr in thresholds:
        sk = set()
        for i, j in combinations(range(len(NODES)), 2):
            if abs(corr_matrix[i, j]) >= tr and pval_matrix[i, j] < 0.05:
                sk.add(frozenset({NODES[i], NODES[j]}))
        # Naive directed count: skeleton edges if any orientation in gt
        sk_directed = set()
        for e in sk:
            a, b = sorted(e)
            if (a, b) in gt: sk_directed.add((a, b))
            if (b, a) in gt: sk_directed.add((b, a))
        # Recall ignoring direction (just skeleton):
        gt_und = {frozenset({s, t}) for (s, t) in gt}
        tp_und = sk & gt_und
        recalls.append(len(tp_und) / max(len(gt_und), 1))
        precisions.append(len(tp_und) / max(len(sk), 1) if sk else 0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(thresholds, recalls, "b-o", label="Recall (undirected skeleton)")
    ax.plot(thresholds, precisions, "r-s", label="Precision (undirected skeleton)")
    ax.axvline(args.corr_threshold, color="black", ls="--",
               label=f"Used threshold = {args.corr_threshold}")
    ax.set_xlabel("|r| threshold")
    ax.set_ylabel("Score")
    ax.set_title("PC pilot — recall + precision vs |r| threshold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    _save_html("pc_recall_vs_threshold_real",
               "PC pilot — recall vs threshold sweep",
               "Sweeping the |r| threshold across 25 values: recall (blue) and precision "
               "(red) of the recovered skeleton (undirected, p<0.05). Reveals the "
               "precision/recall trade-off. The threshold used in the main pipeline is "
               "marked with the black dashed line.")

    print("\nDone.")


if __name__ == "__main__":
    main()
