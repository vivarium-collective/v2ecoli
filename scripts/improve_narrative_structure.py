"""Improve the investigation report's reader-path:
  (1) prepend a one-paragraph "what this is" to the executive summary
  (2) add a "story of the six phases" bridge to the at_a_glance section
  (3) add a plain-English "Why this step matters" line to each study card

Per expert feedback: the report currently jumps from "PDMP is the right
class" to large six-phase machinery without orienting the reader on why
each phase exists. These edits give them a reader's path.

Run from worktree root:
    python scripts/improve_narrative_structure.py
"""
from __future__ import annotations
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


WHAT_THIS_IS = (
    "**What this is:** This report is a roadmap for turning v2ecoli from a successful but "
    "algorithmically assembled whole-cell simulation into a cleaner mathematical model. "
    "The goal is not to replace the biology, but to preserve what v2ecoli already knows "
    "while making the model easier to analyze, fit to data, compile, and use for causal "
    "inference.\n\n"
)


PHASE_STORY = (
    "**The story of the six phases.** The phases move from diagnosis to replacement to "
    "inference. Phase 0 defines what v2ecoli currently does. Phases 1 and 2 replace the "
    "two major dynamical layers: metabolism and stochastic events. Phase 3 adds likelihoods "
    "so the model can be fit to data. Phase 4 makes the model fast enough to run many "
    "trajectories. Phase 5 uses that machinery for causal gene-function discovery.\n\n"
    "Each downstream phase assumes upstream phases have passed their primary tests — the "
    "tests on each study card are the contract."
)


# Plain-English "Why this step matters" per study — goes into study_card.why_before_next
# (renders as a "Why before next" row in the Study Card panel).
PER_STUDY_WHY = {
    "pdmp-00-characterization": (
        "Before changing the model, we need to measure what the current model passes "
        "between subprocesses, so every later replacement can be judged against a "
        "reference."
    ),
    "pdmp-01-metabolism-ode": (
        "This tests whether metabolism can be represented as continuous biochemical "
        "dynamics instead of an FBA optimization step, without breaking the rest of the "
        "cell model."
    ),
    "pdmp-02-jump-processes": (
        "This converts random biological events — transcription, translation, division — "
        "from timestep-based draws into event-based stochastic dynamics."
    ),
    "pdmp-03-inference": (
        "This asks whether the new model can assign probabilities to observations, which "
        "is what makes parameter fitting and Bayesian inference possible."
    ),
    "pdmp-04-compilation": (
        "This asks whether the cleaner mathematical structure can be compiled into a much "
        "faster simulator for large ensembles of cells."
    ),
    "pdmp-05-causal-discovery": (
        "This is the payoff phase: use the fast, likelihood-bearing model to compare "
        "gene-function hypotheses and choose informative perturbations."
    ),
}


def edit_investigation():
    yaml_path = Path("investigations/v2ecoli-pdmp/investigation.yaml")
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # 1. Prepend "What this is" to the lead.
    lead = (spec.get("lead") or "").lstrip()
    if not lead.startswith("**What this is:**"):
        spec["lead"] = WHAT_THIS_IS + lead
        print("  + prepended 'What this is' to investigation.lead")

    # 2. Add the phase-story paragraph before at_a_glance items.
    # Convention: use the `how_to_read` field for evaluator-orientation prose.
    how_to_read = (spec.get("how_to_read") or "").strip()
    if "story of the six phases" not in how_to_read.lower():
        spec["how_to_read"] = PHASE_STORY + (("\n\n" + how_to_read) if how_to_read else "")
        print("  + added 'story of the six phases' bridge to investigation.how_to_read")

    yaml_path.write_text(
        yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def edit_studies():
    for slug, why in PER_STUDY_WHY.items():
        yaml_path = Path("studies") / slug / "study.yaml"
        if not yaml_path.exists():
            print(f"  SKIP {slug}: yaml missing"); continue
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # Some studies have study_card as a plain STRING; the dashboard
        # renderer expects a dict with {goal, mechanism, why_before_next,
        # expected_result, main_expert_question}. Promote string -> dict
        # by routing the prose to `goal` and adding `why_before_next`.
        cur_sc = spec.get("study_card")
        if isinstance(cur_sc, str):
            sc = {"goal": cur_sc.strip()}
        elif isinstance(cur_sc, dict):
            sc = dict(cur_sc)
        else:
            sc = {}

        cur_why = (sc.get("why_before_next") or "").strip()
        if cur_why and cur_why != why:
            print(f"  {slug}: keeping existing why_before_next ({len(cur_why)} chars)")
            continue
        sc["why_before_next"] = why
        spec["study_card"] = sc
        yaml_path.write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        print(f"  {slug}: study_card promoted to dict + why_before_next set ({len(why)} chars)")


def main():
    print("== investigation narrative ==")
    edit_investigation()
    print()
    print("== per-study plain-English why ==")
    edit_studies()


if __name__ == "__main__":
    main()
