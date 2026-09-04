#!/usr/bin/env python
"""Control-panel runner for genotype-01-control-panel.

Per gene: generate the KO bundle -> grade the build-integrity card structurally
(fit-free, ~seconds) -> run the fast ParCa (plain first; on failure retry with
--allow-partial-fit to capture the partial state) -> record expression deltas vs
the shared WT arm. Appends per-gene records to data/panel_summary.json after
every gene, so a crash loses at most one genotype.

A ParCa failure is a RESULT: it is recorded with its failure mode and the run
continues. The runner only exits non-zero on infrastructure failure (a gene with
no recorded outcome at all).

Run from the workspace root (the canonical_runs contract):
    python workspace/studies/genotype-01-control-panel/sims/run_panel.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WS_ROOT = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]
OUT = WS_ROOT / "out" / "genotype-01"
SUMMARY = STUDY_DIR / "data" / "panel_summary.json"
PARCA_CLI = WS_ROOT / "v2ecoli" / "cli" / "parca.py"

# (symbol, gene_id, class, one-line wiring evidence) — classifications verified
# against the model's functional wiring 2026-08-17; see study.yaml's panel block.
PANEL = [
    ("ygiB", "EG11164", "A", "expressed, functionally unwired (y-gene)"),
    ("cspA", "EG10166", "A", "expressed, functionally unwired (cold shock)"),
    ("pgi",  "EG10702", "B", "metabolic catalyst via complex"),
    ("tpiA", "EG11015", "B", "metabolic catalyst via complex"),
    ("cheY", "EG10150", "B", "biologically minor; model-unwired (chemotaxis)"),
    ("rpoB", "EG10894", "C", "RNAP core (mg.RNAP_subunits)"),
    ("rplB", "EG10865", "C", "50S r-protein (mg.ribosomal_proteins)"),
    ("dnaG", "EG10239", "C", "replisome monomer subunit (directly consumed)"),
]


def run_parca(outdir: Path, manifest: Path | None, extra: list[str] = []) -> dict:
    cmd = [sys.executable, str(PARCA_CLI), "--mode", "fast", "--cpus", "8",
           "-o", str(outdir)] + (
        ["--bundle-manifest-path", str(manifest)] if manifest else []) + extra
    proc = subprocess.run(cmd, cwd=WS_ROOT, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    return {"exit": proc.returncode, "tail": tail,
            "state_written": (outdir / "parca_state.pkl").is_file()}


def load_expression(state_path: Path):
    from v2ecoli.processes.parca.data_loader import load_parca_state
    state = load_parca_state(state_path)
    expr = np.asarray(
        state["cell_specs"]["basal"]["fit_cistron_expression"], float).ravel()
    fit_status = state.get("mechanistic_fit_status")
    return expr, ({k: str(v) for k, v in fit_status.items()}
                  if isinstance(fit_status, dict) else str(fit_status))


def expression_delta(wt: np.ndarray, ko: np.ndarray,
                     target_ix: np.ndarray) -> dict:
    # The KO state's fitted expression vector OMITS the deleted cistron(s)
    # (functional absence), so it is shorter than the WT vector by exactly the
    # target count with order otherwise preserved. Align by deleting the target
    # indices from the WT vector; equal-length inputs are compared with the
    # target masked instead.
    if wt.shape != ko.shape:
        if wt.size - ko.size == target_ix.size:
            wt = np.delete(wt, target_ix)
            target_ix = np.empty(0, int)
        else:
            return {"error": f"shape mismatch {wt.shape} vs {ko.shape}"}
    keep = np.ones(wt.shape, bool)
    if target_ix.size:
        keep[target_ix] = False
    both = keep & (np.maximum(wt, ko) > 0)
    rel = np.abs(wt[both] - ko[both]) / np.maximum(wt[both], ko[both])
    return {"n": int(both.sum()),
            "median_rel": float(np.median(rel)),
            "max_rel": float(rel.max()),
            "frac_gt_1pct": float((rel > 0.01).mean()),
            "frac_gt_5pct": float((rel > 0.05).mean())}


def main() -> int:
    from v2ecoli.library import genotype_build as gb
    from v2ecoli.library.report_card import grade_card

    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def flush():
        SUMMARY.write_text(json.dumps({"panel": records}, indent=2) + "\n")

    # --- WT reference arm -------------------------------------------------
    wt_dir = OUT / "wt"
    wt_state = wt_dir / "parca_state.pkl"
    if not wt_state.is_file():
        print("== WT reference arm ==", flush=True)
        wt_run = run_parca(wt_dir, None)
        if wt_run["exit"] != 0 or not wt_run["state_written"]:
            print("WT arm failed — cannot compute any contrast:", wt_run["tail"])
            return 2
    wt_expr, wt_fit = load_expression(wt_state)

    # cistron -> gene mapping for excluding the target from the delta
    wt_raw = gb.resolve_raw_data(None)
    from v2ecoli.processes.parca.data_loader import load_parca_state
    sd = load_parca_state(wt_state)["sim_data_root"]
    gene_of_cistron = np.asarray(sd.process.transcription.cistron_data["gene_id"])

    for sym, gid, cls, wiring in PANEL:
        print(f"== {sym} ({gid}, class {cls}) ==", flush=True)
        gdir = OUT / sym
        rec: dict = {"gene": sym, "gene_id": gid, "class": cls, "wiring": wiring}
        try:
            # 1. bundle + structural card (fit-free)
            manifest, genotype_id, spans = gb.make_knockout_bundle([gid], gdir)
            rec["genotype_id"] = genotype_id
            card, reference = gb.build([gid], workdir=gdir)
            report = grade_card(card, reference)
            axes = {name: ax.get("verdict")
                    for name, ax in report["axes"].items()}
            structural = {k: v for k, v in axes.items() if not k.startswith("fit.")}
            rec["card"] = {"overall_structural_ok":
                           all(v == "within_tol" for v in structural.values()),
                           "axes": axes}

            # 2. ParCa, plain first; failure is a result
            run = run_parca(gdir, manifest)
            rec["parca"] = {"exit": run["exit"],
                            "state_written": run["state_written"]}
            if run["exit"] != 0:
                rec["parca"]["failure_tail"] = run["tail"]
                retry = run_parca(gdir, manifest, ["--allow-partial-fit"])
                rec["parca"]["allow_partial_fit_retry"] = {
                    "exit": retry["exit"],
                    "state_written": retry["state_written"],
                    "tail": None if retry["exit"] == 0 else retry["tail"]}

            # 3. fit status + expression delta, when a state exists
            if (gdir / "parca_state.pkl").is_file():
                ko_expr, ko_fit = load_expression(gdir / "parca_state.pkl")
                rec["mechanistic_fit_status"] = ko_fit
                target_ix = np.where(gene_of_cistron == gid)[0]
                rec["expression_delta_vs_wt"] = expression_delta(
                    wt_expr, ko_expr, target_ix)
        except Exception as exc:  # infrastructure failure for THIS gene
            rec["runner_error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
        flush()

    # exit non-zero only if a gene has no outcome at all
    missing = [r["gene"] for r in records
               if "runner_error" in r or "parca" not in r]
    print(f"\npanel complete: {len(records)} genes, "
          f"{len(missing)} without outcome {missing or ''}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
