#!/usr/bin/env python
"""Route-A runner for genotype-06-trpr-regulon — reproduce the ParCa build failure.

Route A deletes trpR from the chromosome BEFORE reconstruction. It is reported
not to build, failing on the TRP[c] amino-acid homeostasis check. Reproducing
that formally is the point of this script: the failure IS the study's result for
this route, so it is captured with its step, its assertion text and the measured
quantity rather than routed around.

Two genotypes are built, because a trpR-only arm is not like-for-like against a
dtrpR dtnaA reference strain (tnaA is the main catabolic sink for tryptophan):

    trpR       EG11029
    trpRtnaA   EG11029 + EG11005

Each arm: generate the KO bundle from ecoli-sources (NOT read from a scratch
dir -- the bundle is regenerated so this reproduces on any machine), grade the
build-integrity card structurally, then run ParCa and record what happened.

A non-zero ParCa exit is the EXPECTED outcome here and is recorded as a result.
The script exits non-zero only on infrastructure failure -- an arm with no
recorded outcome at all.

Run from the workspace root (the canonical_runs contract):
    python workspace/studies/genotype-06-trpr-regulon/sims/run_route_a.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]
OUT = WS_ROOT / "out" / "genotype-06" / "route-a"
SUMMARY = STUDY_DIR / "data" / "route_a_summary.json"
PARCA_CLI = WS_ROOT / "v2ecoli" / "cli" / "parca.py"

ARMS = [
    ("trpR", ["EG11029"], "the repressor alone"),
    ("trpRtnaA", ["EG11029", "EG11005"], "repressor + the main catabolic sink"),
]

# The failure is reported at the amino-acid homeostasis check. Pull the step and
# the numbers out of the traceback rather than eyeballing them, so the recorded
# evidence is the process's own words.
STEP_RE = re.compile(r"step[_ ]?(\d+)", re.I)
NUM_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def run_parca(outdir: Path, manifest: Path, extra: list[str] = []) -> dict:
    cmd = [sys.executable, str(PARCA_CLI), "--mode", "fast", "--cpus", "8",
           "-o", str(outdir), "--bundle-manifest-path", str(manifest)] + extra
    proc = subprocess.run(cmd, cwd=WS_ROOT, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    lines = combined.strip().splitlines()
    return {
        "exit": proc.returncode,
        "state_written": (outdir / "parca_state.pkl").is_file(),
        "tail": lines[-25:],
        "checkpoints": sorted(p.name for p in outdir.glob("checkpoint_step_*.pkl")),
    }


def classify_failure(run: dict) -> dict:
    """Extract the failure's identity from the traceback: which step, which check."""
    text = "\n".join(run["tail"])
    # Last completed checkpoint bounds where it died: step N written => failed in N+1.
    steps_done = [int(m.group(1)) for c in run["checkpoints"]
                  for m in [STEP_RE.search(c)] if m]
    last_done = max(steps_done) if steps_done else None
    assertion = next((ln.strip() for ln in reversed(run["tail"])
                      if any(k in ln for k in ("Error", "error:", "assert", "Assertion"))), None)
    trp_lines = [ln.strip() for ln in run["tail"] if "TRP" in ln]
    return {
        "last_checkpoint_step": last_done,
        "failed_in_step": (last_done + 1) if last_done is not None else None,
        "assertion_line": assertion,
        "trp_lines": trp_lines,
        "numbers_in_trp_lines": [NUM_RE.findall(ln) for ln in trp_lines] or None,
    }


def main() -> int:
    from v2ecoli.library import genotype_build as gb
    from v2ecoli.library.report_card import grade_card

    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def flush():
        SUMMARY.write_text(json.dumps({"route_a": records}, indent=2) + "\n")

    for name, gene_ids, note in ARMS:
        print(f"== route A: {name} ({', '.join(gene_ids)}) — {note} ==", flush=True)
        adir = OUT / name
        rec: dict = {"arm": name, "gene_ids": gene_ids, "note": note}
        try:
            # 1. Regenerate the bundle. Deliberately not reusing any scratch dir.
            manifest, genotype_id, spans = gb.make_knockout_bundle(gene_ids, adir)
            rec["genotype_id"] = genotype_id
            # Record spans verbatim; their shape is the generator's business,
            # not this script's, and coercing it once cost a whole run.
            rec["deleted_spans"] = json.loads(json.dumps(spans, default=str))

            # 2. Structural build integrity, fit-free. Separates "the genome
            #    splice is sound" from "the fit converged" — the whole point of
            #    recording this failure rather than calling the genotype broken.
            card, reference = gb.build(gene_ids, workdir=adir)
            report = grade_card(card, reference)
            axes = {k: ax.get("verdict") for k, ax in report["axes"].items()}
            structural = {k: v for k, v in axes.items() if not k.startswith("fit.")}
            rec["card"] = {
                "structural_axes_all_ok": all(v == "within_tol" for v in structural.values()),
                "axes": axes,
            }

            # 3. ParCa. A non-zero exit is the expected result for this route.
            run = run_parca(adir, manifest)
            rec["parca"] = {"exit": run["exit"], "state_written": run["state_written"],
                            "checkpoints": run["checkpoints"], "tail": run["tail"]}
            rec["failure"] = classify_failure(run) if run["exit"] != 0 else None
            if run["exit"] == 0:
                # Worth flagging loudly: the study PREDICTS this fails. If it
                # builds, F-05 is wrong and that is the finding.
                rec["unexpected_success"] = (
                    "Route A COMPLETED. F-05 predicted a step-9 TRP[c] failure; "
                    "this contradicts it and the finding must be rewritten.")
        except Exception as exc:
            rec["runner_error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
        flush()

    missing = [r["arm"] for r in records if "runner_error" in r or "parca" not in r]
    print(f"\nroute A complete: {len(records)} arms, {len(missing)} without outcome {missing or ''}")
    for r in records:
        if r.get("parca"):
            f = r.get("failure") or {}
            print(f"  {r['arm']:<10} exit={r['parca']['exit']} "
                  f"failed_in_step={f.get('failed_in_step')} "
                  f"assertion={(f.get('assertion_line') or '')[:70]}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
