"""Add the missing planned_runs[] + simulation_set[] entries for placeholder
viz that currently reference runs not in any study's plan.

Per audit (see chat), 6 placeholder viz reference runs that don't exist in
any study's planned_runs[]. This script adds them as proper entries so the
plan covers every viz the report displays.

Also adds the recently-completed Millard 10000s steady-state approach run
to pdmp-01 as status: ran, since the 5 new millard_real_* viz files are
all generated from out/trajectories/millard_steady_approach_10000s.csv.

Idempotent — uses run name as key.

Run from worktree root:
    python scripts/wire_missing_runs.py
"""
from __future__ import annotations
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


# (study_slug, [planned_run_dict, ...], [simulation_set_dict, ...])
ADDITIONS = {
    "pdmp-00-characterization": {
        "planned_runs": [
            {
                "name": "phase0-profile-instrumented-baseline",
                "status": "planned",
                "n_steps": 1000,
                "details": (
                    "Profile-instrumented single-cell baseline. Wraps every process invocation "
                    "with timeit + tracemalloc so per-step compute is decomposed into "
                    "FBA-LP / RNG-draw / store-marshalling / topology-traversal / emitter-I/O "
                    "buckets. Produces reports/figures/pdmp-00/profile_decomposition_bars.html "
                    "with real means ± std over ≥1000 step samples."
                ),
            },
        ],
        "simulation_set": [
            {
                "name": "phase0-profile-instrumented-baseline",
                "kind": "single",
                "status": "planned",
                "base_model": "v2ecoli.composites.baseline",
                "duration_steps": 1000,
                "seeds": [0],
                "metrics": ["per_step_wall_ms_by_bucket", "tracemalloc_peak"],
                "pass_fail_tests": ["marshalling-overhead-dominates"],
                "details": (
                    "Produces the per-step compute decomposition viz. Required by the "
                    "marshalling-overhead-dominates primary test in this study's tests[]."
                ),
            },
        ],
    },
    "pdmp-01-metabolism-ode": {
        "planned_runs": [
            {
                "name": "millard-steady-state-10000s",
                "status": "ran",
                "n_steps": 500,
                "details": (
                    "10,000 s standalone Millard 2017 ODE via basico (LSODA). 500 sampled "
                    "timepoints. CSV at out/trajectories/millard_steady_approach_10000s.csv. "
                    "Source for millard_real_4panel_trajectory, millard_real_energy_charge, "
                    "millard_real_phase_plane_atp_pep, millard_real_full_species_heatmap, "
                    "millard_real_redox_balance viz. Wall: 0.1 s."
                ),
            },
        ],
        "simulation_set": [
            {
                "name": "millard-steady-state-10000s",
                "kind": "single",
                "status": "completed",
                "base_model": "v2ecoli.composites.millard2017_metabolism",
                "duration_steps": 500,
                "seeds": [0],
                "metrics": ["species_concentrations(t)", "AEC(t)", "NAD/NADH(t)", "NADPH/NADP(t)"],
                "pass_fail_tests": ["millard-published-ref-state-reproduced"],
                "details": (
                    "Completed 2026-05-25. Trajectory drives the 5 millard_real_* viz that "
                    "replace the kinetic_constraint_curves placeholder."
                ),
            },
        ],
    },
    "pdmp-02-jump-processes": {
        "planned_runs": [
            {
                "name": "jump-process-multigen-100-divisions",
                "status": "planned",
                "n_steps": "(multi-gen — ≥100 divisions per replicate)",
                "details": (
                    "Multi-generation jump-process pilot. Produces the inheritance "
                    "distribution + division-time-distribution viz from real division events. "
                    "≥100 divisions per replicate to get reliable variance estimates on "
                    "daughter1−daughter2 binomial-partition test."
                ),
            },
            {
                "name": "jump-process-event-rate-profile",
                "status": "planned",
                "n_steps": 1000,
                "details": (
                    "Event-counting profile run. Increment a counter per reaction firing so we "
                    "can report per-cell event rates spanning translation init (~10² /s) down to "
                    "cell division (~10⁻³ /s). Produces phase2_event_rates viz with real data."
                ),
            },
        ],
        "simulation_set": [
            {
                "name": "jump-process-multigen-100-divisions",
                "kind": "single",
                "status": "planned",
                "base_model": "v2ecoli.composites.jump_multigen",
                "duration_steps": "(multi-gen)",
                "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
                "metrics": ["division_time", "daughter_partition_per_protein", "inherited_copy_distribution"],
                "pass_fail_tests": ["multi-gen-inheritance-binomial"],
                "details": (
                    "Required by multi-gen-inheritance-binomial primary test. Produces "
                    "inheritance_distribution_v2 + division_time_distribution viz."
                ),
            },
            {
                "name": "jump-process-event-rate-profile",
                "kind": "single",
                "status": "planned",
                "base_model": "v2ecoli.composites.jump_transcription_pilot",
                "duration_steps": 1000,
                "seeds": [0],
                "metrics": ["event_rate_per_reaction", "total_events"],
                "pass_fail_tests": ["jump-process-likelihood-closed-form"],
                "details": (
                    "Event rate measurements span 6+ OOM — motivates τ-leap for high-rate "
                    "reactions. Produces phase2_event_rates viz."
                ),
            },
        ],
    },
    "pdmp-04-compilation": {
        "planned_runs": [
            {
                "name": "jax-jit-cold-warm-bench",
                "status": "planned",
                "n_steps": "(20 sequential single-step runs)",
                "details": (
                    "JAX backend cold-start cost amortization bench. Run 20 sequential "
                    "single-trajectory invocations; the FIRST pays JIT compilation, runs 2-20 "
                    "are warm-cache. Compares wall time per run. Produces jit_compile_cost viz."
                ),
            },
            {
                "name": "ram-profile-per-cell-N1024",
                "status": "planned",
                "n_steps": 600,
                "details": (
                    "RSS profile of N=1024 cells × 600 steps using tracemalloc + psutil. "
                    "Decomposes RAM into bulk[], unique[], process_state, listeners, emitter "
                    "buffer. Produces memory_footprint viz with real measurements."
                ),
            },
        ],
        "simulation_set": [
            {
                "name": "jax-jit-cold-warm-bench",
                "kind": "sweep",
                "status": "planned",
                "base_model": "v2ecoli.compiled.jax_pdmp",
                "axes": [{"parameter": "run_index", "values": list(range(20))}],
                "metrics": ["wall_seconds_per_run", "compile_seconds_first"],
                "pass_fail_tests": [],
                "details": "JIT amortization measurement. Produces jit_compile_cost viz.",
            },
            {
                "name": "ram-profile-per-cell-N1024",
                "kind": "single",
                "status": "planned",
                "base_model": "v2ecoli.composites.baseline_ensemble",
                "duration_steps": 600,
                "seeds": list(range(8)),
                "metrics": ["RSS_GB", "RAM_by_bucket_MB"],
                "pass_fail_tests": ["memory-fits-1024-cells-per-node"],
                "details": (
                    "Required by memory-fits-1024-cells-per-node supporting test. Produces "
                    "memory_footprint viz with real per-bucket RAM."
                ),
            },
        ],
    },
}


def merge_unique(existing: list, additions: list, key: str = "name") -> tuple[list, int]:
    """Append additions whose key isn't already in existing. Returns (merged, n_added)."""
    have = {e.get(key) for e in (existing or []) if isinstance(e, dict)}
    out = list(existing or [])
    added = 0
    for a in additions:
        if a.get(key) in have:
            continue
        out.append(a)
        added += 1
    return out, added


def main():
    for slug, blocks in ADDITIONS.items():
        yaml_path = Path("workspace/studies") / slug / "study.yaml"
        if not yaml_path.exists():
            print(f"  SKIP {slug}: yaml missing"); continue
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        pr_after, pr_added = merge_unique(spec.get("planned_runs"), blocks.get("planned_runs", []))
        sim_after, sim_added = merge_unique(spec.get("simulation_set"), blocks.get("simulation_set", []))

        if pr_added == 0 and sim_added == 0:
            print(f"  {slug}: nothing to add (already present)")
            continue
        spec["planned_runs"] = pr_after
        spec["simulation_set"] = sim_after
        yaml_path.write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        print(f"  {slug}: +{pr_added} planned_runs  +{sim_added} simulation_set entries")


if __name__ == "__main__":
    main()
