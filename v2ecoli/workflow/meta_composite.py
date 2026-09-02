"""Build the meta-composite document for a variants × seeds × generations sweep.

One LineageProcess node per (variant, seed) branch lives under
``state.branches[<branch-key>]``. The whole sweep is a single process-bigraph
document, saveable/inspectable via v2ecoli.pbg.save_pbg_doc.
"""

from __future__ import annotations

from typing import Any

from v2ecoli.workflow.variants import expand_branches, BranchSpec


def register_workflow_processes(core) -> None:
    """Register workflow Processes/Steps so ``local:`` addresses resolve."""
    from v2ecoli.workflow.lineage import LineageProcess
    core.register_link("LineageProcess", LineageProcess)


def _branch_key(spec: BranchSpec) -> str:
    return f"variant={spec.variant_index}/seed={spec.seed}"


def _lineage_node(spec: BranchSpec, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lineage": {
            "_type": "process",
            "address": "local:LineageProcess",
            "interval": float(config.get("time_step", 1.0)),  # Composite tick cadence, not the internal sim step
            "config": {
                # NOTE: keep these keys in sync with LineageProcess.config_schema.
                "cache_dir": config.get("cache_dir", "out/cache"),
                "seed": spec.seed,
                "lineage_seed": spec.seed,  # same as seed in MVP: one lineage per seed
                "variant_index": spec.variant_index,
                "variant_name": spec.variant_name,
                "config_overrides": dict(spec.overrides),
                "generations": int(config.get("generations", 1)),
                "single_daughters": bool(config.get("single_daughters", True)),
                # Per-generation checkpoint/resume (backlog item 34) — see
                # LineageProcess.config_schema.
                "initial_carry_state_path": config.get("initial_carry_state_path", ""),
                "initial_generation_index": int(config.get("initial_generation_index", 0)),
                "daughter_state_out_path": config.get("daughter_state_out_path", ""),
                "experiment_id": config.get("experiment_id", "default"),
                "out_dir": config.get("out_dir", "out/workflow"),
                "max_duration_per_gen": float(config.get("max_duration_per_gen", 3600.0)),
                "time_step": float(config.get("time_step", 1.0)),
                "media": config.get("media", "minimal"),
                "emitter": config.get("emitter", "parquet"),
                "emitter_arg": dict(config.get("emitter_arg") or {}),
                "injected_processes": config.get("injected_processes") or {},
                # Per-cell biological build kwargs — forwarded so every
                # generation's baseline() build engages the SAME biology as the
                # single-cell path (audit: batch mode dropped these, degrading an
                # injected metabolism-redux/violacein batch to basal FBA).
                # LineageProcess.config_schema supplies the defaults when absent.
                "features": list(config.get("features") or []),
                "ppgpp_regulation": bool(config.get("ppgpp_regulation", True)),
                "trna_attenuation": bool(config.get("trna_attenuation", False)),
                "supercoiling": bool(config.get("supercoiling", False)),
                "mass_conservation": bool(config.get("mass_conservation", False)),
                "exchange_fluxes": dict(config.get("exchange_fluxes") or {}),
                "exchange_flux_basis": config.get("exchange_flux_basis") or "",
                "transcript_initiation_mode": (
                    config.get("transcript_initiation_mode") or "discrete"),
                "polypeptide_initiation_mode": (
                    config.get("polypeptide_initiation_mode") or "discrete"),
            },
            "inputs": {},
            "outputs": {
                "summary": ["summary"],
                "complete": ["complete"],
            },
        },
        "summary": {},
        "complete": False,
    }


def build_meta_composite(config: dict[str, Any]) -> dict[str, Any]:
    """Return a process-bigraph document for the full sweep described by ``config``."""
    branches = expand_branches(config)
    branch_state = {_branch_key(spec): _lineage_node(spec, config) for spec in branches}
    return {
        "state": {
            "global_time": 0.0,
            "branches": branch_state,
        },
        "skip_initial_steps": True,
        "sequential_steps": False,
    }
