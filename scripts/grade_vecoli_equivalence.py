"""Grade a v2ecoli ensemble against a pinned vEcoli ("v1") equivalence reference.

The sibling of ``pin_vecoli_equivalence_reference.py``: that script measures the
**reference** (v1) side and bakes it; this one measures the **candidate** (v2)
side from its own sweep, grades the two, and writes the
``report_card_verdict.json`` that ``VsVecoliCard`` renders.

Until now nothing in the repo joined those two halves — ``grade_card`` was called
only from tests, and the committed basal verdict was emitted by hand during
unrelated work. That is why the committed verdict and the committed reference
came to describe different ensembles without anything noticing.

**Any nutrient condition** (``--condition``, default ``basal``), for the same
reason the pin script is: nothing in the measurement branches on the condition,
so five conditions are five invocations rather than five scripts. ``--condition``
selects the card directory, which is where both the reference is read from and
the verdict is written.

The card stays **render-only** — this writes a verdict, it does not add anything
to ``GRADED``. Promoting an axis to a gate is a separate decision (see #439).

Measurement (all of it already exists; this script only joins it):
  * ``card_from_analysis`` — the per-cell scalar axes, out of the sweep's
    ``analysis.json`` (so the candidate sweep must have been through the
    analysis runner, not merely left as parquet).
  * ``merge_vectors``      — omics + exchange-flux axes, read from the sweep
    parquet, with named flux KPIs sliced by the ``flux_ids`` order pinned in the
    reference. Heavy (~minute).
  * ``grade_card`` / ``verdict_json`` — the shared grader and the v1 verdict
    schema, unchanged.

Provenance: the emitted verdict records **which reference it graded against** —
the reference's model ref, sweep and per-axis n — because the previous verdict
recorded none of that, and so could disagree with its own reference silently.

Run:
    python scripts/grade_vecoli_equivalence.py \\
        --sweep-dir out/population_phenotype_basal \\
        --condition basal --model-ref <v2ecoli git SHA> --gen-lb 3
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from v2ecoli.library.report_card import (
    card_from_analysis,
    grade_card,
    merge_vectors,
    verdict_json,
)

_CARD_ROOT = os.path.join("docs", "report_cards")


def _card_dir(condition: str) -> str:
    """The vs_vecoli directory of the card named for ``condition`` — the same
    location ``pin_vecoli_equivalence_reference._default_out`` writes into."""
    return os.path.join(_CARD_ROOT, f"population_phenotype_{condition}", "vs_vecoli")


def _default_reference(condition: str) -> str:
    return os.path.join(_card_dir(condition), "vecoli_reference.json")


def _default_out(condition: str) -> str:
    return os.path.join(_card_dir(condition), "report_card_verdict.json")


def _reference_provenance(reference: dict) -> dict:
    """What the verdict must record about the reference it graded against.

    The pre-existing basal verdict carried `reference_model` and nothing else —
    no reference commit, no sweep, no n — so it could describe a different
    ensemble than the reference file beside it and nothing could surface the
    disagreement. Carry the reference's own stimulus stamp plus the per-axis n
    actually used, so a reader can tell what was compared.
    """
    stim = reference.get("stimulus") or {}
    n_by_axis = {}
    for path, spec in (reference.get("axes") or {}).items():
        ref_values = (spec.get("criterion") or {}).get("ref_values")
        if isinstance(ref_values, list):
            n_by_axis[path] = len(ref_values)
    prov = {
        "reference_model_ref": stim.get("blessed_model_ref"),
        "reference_sweep_dir": stim.get("sweep_dir"),
        "reference_ensemble": stim.get("ensemble"),
        "reference_condition": stim.get("condition"),
        "reference_generation_lower_bound": stim.get("generation_lower_bound"),
    }
    if n_by_axis:
        prov["reference_n_by_axis"] = n_by_axis
        prov["reference_n_min"] = min(n_by_axis.values())
        prov["reference_n_max"] = max(n_by_axis.values())
    return prov


def _candidate_provenance(card: dict, sweep_dir: str, analysis_path: str,
                          gen_lb: int) -> dict:
    """The same, for the measured side."""
    prov = {
        "candidate_sweep_dir": sweep_dir,
        "candidate_analysis": analysis_path,
        "candidate_generation_lower_bound": gen_lb,
    }
    n_cells = card.get("n_cells")
    if n_cells is not None:
        prov["candidate_n_cells"] = n_cells
    health = card.get("sim_health")
    if isinstance(health, dict):
        # a run-quality readout: cells that hit the duration cap without
        # dividing are not a phenotype, and a reader should see how many.
        prov["candidate_sim_health"] = health
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True,
                    help="v2ecoli parquet sweep dir (the CANDIDATE ensemble)")
    ap.add_argument("--condition", default="basal",
                    help="nutrient condition; selects the card dir the reference "
                         "is read from and the verdict is written to")
    ap.add_argument("--analysis", default=None,
                    help="analysis.json holding the multiseed "
                         "population_phenotype_basal result "
                         "(default: <sweep-dir>/analysis.json)")
    ap.add_argument("--reference", default=None,
                    help="pinned v1 reference json (default: the vs_vecoli "
                         "reference of the card named for --condition)")
    ap.add_argument("--model-ref", required=True,
                    help="v2ecoli git commit that produced the candidate sweep "
                         "(provenance)")
    ap.add_argument("--gen-lb", type=int, default=0,
                    help="generation_lower_bound for the VECTOR axes; must match "
                         "the bound the analysis.json scalars were built with")
    ap.add_argument("--out", default=None,
                    help="output verdict json (default: the card's "
                         "report_card_verdict.json)")
    ap.add_argument("--skip-vectors", action="store_true",
                    help="grade only the scalar per-cell axes; the omics/flux "
                         "axes then render ungraded (fast, for a plumbing check)")
    args = ap.parse_args()

    analysis_path = args.analysis or os.path.join(args.sweep_dir, "analysis.json")
    reference_path = args.reference or _default_reference(args.condition)
    out_path = args.out or _default_out(args.condition)

    if not os.path.isfile(analysis_path):
        raise SystemExit(
            f"no analysis.json at {analysis_path}. The candidate sweep must have "
            f"been through the analysis runner — raw parquet alone does not carry "
            f"the per-cell scalar axes.")
    if not os.path.isfile(reference_path):
        raise SystemExit(
            f"no pinned reference at {reference_path}. Pin one first:\n"
            f"  python scripts/pin_vecoli_equivalence_reference.py "
            f"--sweep-dir <v1 sweep> --condition {args.condition} --model-ref <sha>")

    with open(reference_path, encoding="utf-8") as f:
        reference = json.load(f)
    with open(analysis_path, encoding="utf-8") as f:
        card = card_from_analysis(json.load(f))

    ref_cond = (reference.get("stimulus") or {}).get("condition")
    if ref_cond and ref_cond != args.condition:
        raise SystemExit(
            f"reference at {reference_path} is pinned for condition {ref_cond!r}, "
            f"but --condition is {args.condition!r}. Grading a candidate against "
            f"another condition's reference compares two different stimuli.")

    print(f"[reference] {reference_path}")
    print(f"[candidate] {analysis_path}  (n_cells={card.get('n_cells')})")

    if args.skip_vectors:
        print("[vectors] skipped — omics/flux axes will render ungraded")
    else:
        print(f"[vectors] extracting from {args.sweep_dir} (gen >= {args.gen_lb}) …")
        card = merge_vectors(card, reference, args.sweep_dir, args.gen_lb)

    report = grade_card(card, reference)
    vjson = verdict_json(
        report,
        model_ref=args.model_ref,
        reference_model=(reference.get("stimulus") or {}).get("reference_model", ""),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
    )
    vjson["title"] = reference.get("title") or (
        f"vEcoli <-> v2ecoli — population phenotype ({args.condition})")
    vjson["condition"] = args.condition
    vjson["provenance"] = {
        **_reference_provenance(reference),
        **_candidate_provenance(card, args.sweep_dir, analysis_path, args.gen_lb),
    }

    for gslug, g in sorted(vjson["groups"].items()):
        print(f"  {gslug:22} {g['verdict']:12} ({len(g['axes'])} axes)")
    print(f"  {'OVERALL':22} {vjson['overall']}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vjson, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
