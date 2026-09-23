#!/usr/bin/env python
"""Route-B runner for genotype-06-trpr-regulon — multigeneration translation-level KO.

Three arms, each built from a cache so the perturbation survives cell division:

    wt          out/cache                        (unperturbed)
    trpR        out/genotype-06/cache-trpR       (EG11029)
    trpRtnaA    out/genotype-06/cache-trpRtnaA   (EG11029 + EG11005)

⚠ WHY CACHES AND NOT `knockouts=`. Measured 2026-08-21: a build-time
`knockouts=` argument is applied to the mother and LOST in both daughters
(`translation_efficiencies[3834]`: mother 0.0, daughters 2.4992e-04), because
`ecoli_baseline.py:1414-1415` compiles it into `config_overrides` and
`division.py:373-376` does not thread that to the daughter rebuild. The same
measurement against a cache-baked knockout gives 0.0 in BOTH daughters. v2ecoli#505
fixes the argument path; this runner does not depend on it.

⚠ WHY MULTIGENERATION. A translation-level knockout of a repressor is not
instantaneous: zeroing translation stops synthesis, but the existing TrpR pool
does not decay appreciably — measured flat at 135 copies across 300 steps while
the WT arm's grew. Clearance is by growth dilution and division partitioning, so
derepression can only appear over generations. At a 44 min doubling (2640 steps
at 1 s) a sub-generation run shows nothing, which is a property of the lever and
not a failed experiment.

Readouts are recorded as full traces; band windows are computed downstream so the
window rule can be changed without re-running. Nothing here grades anything.

Run from the workspace root:
    python workspace/studies/genotype-06-trpr-regulon/sims/run_route_b.py --arm wt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[4]

# Resolved once from out/cache/simData.cPickle; see the notebook's index table.
TU_TRP = 2635          # TU00067 = trpLEDCBA (all six genes, one TU)
MONO_TRPR = 3834       # PD00423[c]
MONO_TNAA_HINT = "TRYPTOPHAN-MONOMER[c]"

ARMS = {
    "wt":       ("out/cache",                      "unperturbed reference"),
    "trpR":     ("out/genotype-06/cache-trpR",     "EG11029 translation zeroed"),
    "trpRtnaA": ("out/genotype-06/cache-trpRtnaA", "EG11029 + EG11005 zeroed"),
    # AA-replete pair. `minimal` supplies TRP=0.0, `minimal_plus_amino_acids`
    # supplies TRP=0.1 — so these are the arms in which TrpR has its
    # corepressor available and can actually repress. The basal pair above
    # cannot show derepression if there was nothing to derepress, which is the
    # ambiguity these exist to resolve. ⚠ Blunt instrument: all 20 amino acids
    # are supplied (30 species differ from minimal) and the doubling time drops
    # 44 -> 25 min. The condition-level effects hit BOTH arms, so the
    # within-condition WT/KO ratio is still the comparison.
    "wt_withaa":   ("out/genotype-06/cache-wt-withaa",   "unperturbed, AA-replete"),
    "trpR_withaa": ("out/genotype-06/cache-trpR-withaa", "EG11029 zeroed, AA-replete"),
}

GEN_STEPS = 2640       # one doubling at 44 min (basal), 1 s/step
GEN_STEPS_WITHAA = 1500  # one doubling at 25 min (with_aa)
SAMPLE_EVERY = 120


def lineage_agent(agents: dict) -> str:
    """Follow one lineage deterministically: the all-zeros daughter each time."""
    return sorted(agents)[0]


def sample(comp) -> dict | None:
    ags = comp.state["agents"]
    aid = lineage_agent(ags)
    li = ags[aid]["listeners"].get("rna_synth_prob", {})
    tu = np.asarray(li.get("actual_rna_synth_prob", []))
    if tu.size <= TU_TRP:
        return None
    mc = np.asarray(ags[aid]["listeners"].get("monomer_counts", []))
    return {
        "agent": aid,
        "n_agents": len(ags),
        "trp_tu_synth_prob": float(tu[TU_TRP]),
        "total_rna_init": float(np.asarray(li.get("total_rna_init", 0)).sum()),
        "trpR_monomer": float(mc[MONO_TRPR]) if mc.size > MONO_TRPR else None,
    }


def run_arm(arm: str, generations: int, gen_steps: int | None = None) -> dict:
    import v2ecoli
    global GEN_STEPS
    GEN_STEPS = gen_steps or (GEN_STEPS_WITHAA if arm.endswith("_withaa") else 2640)
    cache, note = ARMS[arm]
    print(f"== route B: {arm} ({note}) cache={cache} ==", flush=True)
    t0 = time.time()
    # ⚠ experiment_id MUST be distinct per arm. The ParquetEmitter writes its
    # configuration under .../experiment_id=<id>/generation=N/agent_id=M and
    # calls makedirs WITHOUT exist_ok, so two arms sharing the default id race
    # at the first division and one of them dies with FileExistsError. Cost a
    # 23-minute run of two arms to find.
    comp = v2ecoli.build_composite("ecoli_baseline", seed=0, cache_dir=cache,
                                   experiment_id=f"genotype06-{arm}")

    trace = []
    total = generations * GEN_STEPS
    for step in range(SAMPLE_EVERY, total + 1, SAMPLE_EVERY):
        comp.run(SAMPLE_EVERY)
        rec = sample(comp)
        if rec is not None:
            rec["step"] = step
            trace.append(rec)
            if step % (GEN_STEPS // 2) == 0:
                print(f"  step {step:>6} agents={rec['n_agents']:>2} "
                      f"trpTU={rec['trp_tu_synth_prob']:.4e} "
                      f"TrpR={rec['trpR_monomer']}", flush=True)
    out = {"arm": arm, "cache": cache, "note": note,
           "generations": generations, "gen_steps": GEN_STEPS,
           "sample_every": SAMPLE_EVERY,
           "wall_seconds": round(time.time() - t0, 1), "trace": trace}
    dest = STUDY_DIR / "data" / f"route_b_{arm}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  wrote {dest.name} ({len(trace)} samples, {out['wall_seconds']}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--gen-steps", type=int, default=None,
                    help="override steps per generation (default: 2640 basal / 1500 with_aa)")
    args = ap.parse_args()
    os.chdir(WS_ROOT)
    run_arm(args.arm, args.generations, args.gen_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
