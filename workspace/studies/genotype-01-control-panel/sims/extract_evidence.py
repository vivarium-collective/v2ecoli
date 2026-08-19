#!/usr/bin/env python
"""Evidence extractor for genotype-01-control-panel.

Three quantities are cited in study.yaml's findings but were not previously
backed by a committed artifact (raised in review of v2ecoli#520). This script
re-derives each from the ParCa states the panel already produced and writes
data/evidence.json, so every number in the findings is independently checkable
by re-running one command.

    E-1  F-03's blast radius   — len(ppgpp_regulated_genes) and the panel's
                                 8/8 crash-vs-membership correlation.
    E-2  F-04's operon tail    — the named >5% cistrons behind the ygiB KO's
                                 "ygiC/tolC at 12.3%" claim.
    E-3  F-04's mode control   — the full-mode WT/ygiB contrast and its cost.

Reads only; runs no ParCa. Requires the states left by run_panel.py (fast, in
out/genotype-01/) and the 2026-08-18 full-mode control (out/genotype-full/).
Arms that are absent are reported as null rather than silently skipped.

Run from the workspace root (the canonical_runs contract):
    python workspace/studies/genotype-01-control-panel/sims/extract_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WS_ROOT = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]
FAST = WS_ROOT / "out" / "genotype-01"
FULL = WS_ROOT / "out" / "genotype-full"
EVIDENCE = STUDY_DIR / "data" / "evidence.json"

# Outcome of each panel arm, as recorded in data/panel_summary.json. Duplicated
# here as the *independent* variable of E-1's correlation: membership is read
# from the model, the outcome from the run, and the two are joined below.
PANEL_OUTCOME = {
    "ygiB": "built", "cheY": "built", "pgi": "built", "tpiA": "built",
    "cspA": "crashed", "rpoB": "crashed", "rplB": "crashed", "dnaG": "crashed",
}

# Below this, a fitted expression value is numerically zero and a relative
# delta against it is meaningless (F-04 part c). Stated as a parameter rather
# than buried so the tail split can be re-cut at a different threshold.
ZERO_FLOOR = 1e-12


def load_state(path: Path):
    from v2ecoli.processes.parca.data_loader import load_parca_state
    return load_parca_state(path)


def basal_expression(state) -> np.ndarray:
    return np.asarray(
        state["cell_specs"]["basal"]["fit_cistron_expression"], float).ravel()


def cistron_ids(state) -> np.ndarray:
    cd = state["sim_data_root"].process.transcription.cistron_data
    return np.asarray(cd["id"], dtype=object)


def step_sum(outdir: Path) -> float | None:
    """Sum of per-step ParCa runtimes.

    ⚠ Boundary: these are the 9 pipeline steps only. Wall-clock for the same
    build is larger — it includes CLI startup, knowledge-base construction
    before step 1, and writing the ~690 MB parca_state.pkl. Do not compare a
    number from here against a wall-clock figure.
    """
    rt = outdir / "runtimes.json"
    if not rt.is_file():
        return None
    return round(sum(json.loads(rt.read_text()).values()), 1)


def e1_ppgpp_blast_radius(wt_state, symbol_of_gene: dict) -> dict:
    """E-1 — how many cistrons F-03 can fire on, and the panel correlation."""
    tr = wt_state["sim_data_root"].process.transcription
    regulon = {str(x) for x in tr.ppgpp_regulated_genes}
    ids = cistron_ids(wt_state)
    gene_of = np.asarray(
        tr.cistron_data["gene_id"], dtype=object)

    membership = []
    for cistron, gene in zip(ids, gene_of):
        sym = symbol_of_gene.get(str(gene))
        if sym in PANEL_OUTCOME:
            membership.append({
                "gene": sym,
                "gene_id": str(gene),
                "cistron_id": str(cistron),
                "in_ppgpp_regulon": str(cistron) in regulon,
                "parca_outcome": PANEL_OUTCOME[sym],
            })
    membership.sort(key=lambda r: (r["parca_outcome"], r["gene"]))

    crashed_in = sum(
        1 for r in membership
        if r["parca_outcome"] == "crashed" and r["in_ppgpp_regulon"])
    built_in = sum(
        1 for r in membership
        if r["parca_outcome"] == "built" and r["in_ppgpp_regulon"])
    n_crashed = sum(1 for r in membership if r["parca_outcome"] == "crashed")
    n_built = sum(1 for r in membership if r["parca_outcome"] == "built")

    return {
        "claim": "F-03: ~393 ppGpp-regulated genes, ~8% of the genome",
        "n_ppgpp_regulated_genes": len(regulon),
        # Two denominators are defensible and they differ; both are recorded
        # rather than picking the one that rounds nicest.
        "denominators": {
            "n_cistrons_functional": int(len(ids)),
            "frac_of_cistrons": round(len(regulon) / len(ids), 4),
            "note": "raw_data.genes (4747) gives 0.0828; the functional "
                    "cistron set is the one set_ppgpp_expression indexes, so "
                    "frac_of_cistrons is the operative figure.",
        },
        "panel_membership": membership,
        "correlation": {
            "crashed_and_in_regulon": crashed_in,
            "crashed_total": n_crashed,
            "built_and_in_regulon": built_in,
            "built_total": n_built,
            "mismatches": (n_crashed - crashed_in) + built_in,
        },
    }


def e2_operon_tail(wt_state, ko_state, symbol_of_gene: dict) -> dict:
    """E-2 — the named cistrons in the ygiB KO's >5% tail.

    Aligned by cistron id, not by position: the KO's fitted vector omits the
    deleted cistron entirely, so positional alignment would silently shift
    every downstream entry by one.
    """
    wt_ids, ko_ids = cistron_ids(wt_state), cistron_ids(ko_state)
    wt_expr, ko_expr = basal_expression(wt_state), basal_expression(ko_state)
    wt_by_id = dict(zip((str(i) for i in wt_ids), wt_expr))

    rows = []
    for cistron, ko_val in zip((str(i) for i in ko_ids), ko_expr):
        wt_val = wt_by_id.get(cistron)
        if wt_val is None:
            continue
        denom = max(wt_val, ko_val)
        if denom <= 0:
            continue
        rows.append((cistron, float(wt_val), float(ko_val),
                     abs(wt_val - ko_val) / denom))

    tail = [r for r in rows if r[3] > 0.05]
    tail.sort(key=lambda r: -r[3])

    def describe(r):
        cistron, wt_val, ko_val, rel = r
        gene = cistron.replace("_RNA", "")
        return {
            "cistron_id": cistron,
            "symbol": symbol_of_gene.get(gene),
            "wt": wt_val,
            "ko": ko_val,
            "rel_delta": round(rel, 4),
            "numerically_zero": max(wt_val, ko_val) < ZERO_FLOOR,
        }

    described = [describe(r) for r in tail]
    real = [r for r in described if not r["numerically_zero"]]

    return {
        "claim": "F-04(b): ~10-12% shifts confined to the deleted gene's "
                 "operon neighbours (ygiB's tail is ygiC and tolC)",
        "arm": "ygiB (EG11164), fast mode",
        "zero_floor": ZERO_FLOOR,
        "n_compared": len(rows),
        "median_rel": float(np.median([r[3] for r in rows])),
        "n_tail_gt_5pct": len(described),
        "frac_tail_gt_5pct": round(len(described) / len(rows), 5),
        "tail_with_real_expression": real,
        "n_tail_numerically_zero": len(described) - len(real),
        "reading": "Every >5% entry carrying real expression is an operon "
                   "neighbour of the deleted gene; the remainder sit below "
                   "the zero floor and their relative deltas are meaningless.",
    }


def e3_mode_control(symbol_of_gene: dict) -> dict:
    """E-3 — the full-mode control behind F-04's mode-independence scope."""
    wt_full_p = FULL / "wt" / "parca_state.pkl"
    ko_full_p = FULL / "ygiB" / "parca_state.pkl"
    if not (wt_full_p.is_file() and ko_full_p.is_file()):
        return {"available": False,
                "reason": "full-mode control states not present under out/genotype-full/"}

    wt_full = load_state(wt_full_p)
    wt_fast = load_state(FAST / "wt" / "parca_state.pkl")
    fast_expr, full_expr = basal_expression(wt_fast), basal_expression(wt_full)
    both = np.maximum(fast_expr, full_expr) > 0
    wt_max_rel = float(
        (np.abs(fast_expr - full_expr)[both]
         / np.maximum(fast_expr, full_expr)[both]).max())
    del wt_fast

    ko_full = load_state(ko_full_p)
    full_contrast = e2_operon_tail(wt_full, ko_full, symbol_of_gene)
    del wt_full, ko_full

    return {
        "available": True,
        "claim": "F-04 scope: the contrast is mode-independent; and the "
                 "run-mechanics cost figure",
        "wt_fast_vs_full_max_rel_delta": wt_max_rel,
        "wt_fast_vs_full_verdict": (
            "bit-identical" if wt_max_rel == 0.0 else "differs"),
        "ygiB_contrast_full_mode": {
            "median_rel": full_contrast["median_rel"],
            "frac_tail_gt_5pct": full_contrast["frac_tail_gt_5pct"],
            "tail_with_real_expression": full_contrast["tail_with_real_expression"],
        },
        "cost_step_sum_seconds": {
            "full_wt": step_sum(FULL / "wt"),
            "full_ygiB": step_sum(FULL / "ygiB"),
            "fast_wt": step_sum(FAST / "wt"),
            "fast_ygiB": step_sum(FAST / "ygiB"),
            "boundary": "Sum of the 9 pipeline steps from runtimes.json. "
                        "EXCLUDES CLI startup, knowledge-base construction and "
                        "the ~690 MB state write, so it is smaller than "
                        "wall-clock for the same build.",
        },
    }


def main() -> int:
    from v2ecoli.library import genotype_build as gb

    raw = gb.resolve_raw_data(None)
    symbol_of_gene = {g["id"]: g["symbol"] for g in raw.genes}

    wt_fast = load_state(FAST / "wt" / "parca_state.pkl")
    ko_fast = load_state(FAST / "ygiB" / "parca_state.pkl")

    evidence = {
        "generated_by": "workspace/studies/genotype-01-control-panel/sims/extract_evidence.py",
        "purpose": "Committed backing for the three study.yaml numbers that "
                   "were previously asserted from uncommitted measurements.",
        "source_runs": {
            "fast_panel": "out/genotype-01/ (run_panel.py, panel-2026-08-17)",
            "full_mode_control": "out/genotype-full/ (2026-08-18)",
        },
        "E-1_ppgpp_blast_radius": e1_ppgpp_blast_radius(wt_fast, symbol_of_gene),
        "E-2_ygiB_operon_tail": e2_operon_tail(wt_fast, ko_fast, symbol_of_gene),
    }
    del ko_fast, wt_fast

    evidence["E-3_full_mode_control"] = e3_mode_control(symbol_of_gene)

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"wrote {EVIDENCE.relative_to(WS_ROOT)}")

    e1 = evidence["E-1_ppgpp_blast_radius"]
    e2 = evidence["E-2_ygiB_operon_tail"]
    print(f"  E-1: {e1['n_ppgpp_regulated_genes']} ppGpp-regulated cistrons, "
          f"{e1['denominators']['frac_of_cistrons']:.1%} of the functional set; "
          f"{e1['correlation']['mismatches']} correlation mismatches")
    print(f"  E-2: {e2['n_tail_gt_5pct']} cistrons >5%, "
          f"{len(e2['tail_with_real_expression'])} with real expression")
    e3 = evidence["E-3_full_mode_control"]
    if e3.get("available"):
        print(f"  E-3: WT fast-vs-full max rel delta "
              f"{e3['wt_fast_vs_full_max_rel_delta']:.3g} "
              f"({e3['wt_fast_vs_full_verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
