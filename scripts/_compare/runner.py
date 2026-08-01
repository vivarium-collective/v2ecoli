"""Run a study or investigation from YAML: engines -> render -> verdict -> materialize.

The study-YAML-only execution layer (replaces the manifest-driven run_comparison).
Per study it runs BOTH engines via run_comparison_ensemble.py — v2ecoli with
matched-initial-state on its ParCa cache, genuine vEcoli via vivarium-process on
the upstream cache — into out/<study.name>, then renders the report (which writes
the per-study report_card_verdict.json) and materializes the study's
report_cards/behavior_tests from its cards.
"""
from __future__ import annotations

import subprocess

from scripts._compare.study_spec import REPO, load_investigation, load_study
from scripts._compare.materialize import materialize_study

PER_GEN_STEPS = 15000          # per-generation tick budget (non-binding cap)
PY = str(REPO / ".venv/bin/python")


def _run_engines(spec, out: str, mode: str) -> None:
    """Run both engines for one study into out/<name>. v2ecoli interprets
    --max-steps as a TOTAL across gens; vEcoli interprets it PER-generation."""
    out_c = f"{out}/{spec.name}"
    ref_sd = f"{spec.ve_cache}/simData.cPickle"
    per_gen = spec.max_steps_per_gen        # study-overridable (default 15000)
    v2_cap = str(spec.gens * per_gen)
    # config is the unit: a path drives a process swap on BOTH engines; a bare
    # condition name is a plain baseline comparison (no swap flag).
    is_path = str(spec.config).endswith(".json")
    swap_flags = ["--from-vecoli-config", spec.config] if is_path else []
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "v2ecoli", "--condition", spec.condition,
                    "--cache-dir", spec.v2_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", v2_cap,
                    "--chunk", "60", "--mode", mode,
                    "--match-initial-state", "--match-vecoli-simdata", ref_sd,
                    *swap_flags, "--out-root", out_c], cwd=REPO, check=True)
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "vecoli", "--condition", spec.condition,
                    "--cache-dir", spec.ve_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", str(per_gen),
                    "--chunk", "60", "--mode", mode,
                    "--vecoli-source", "vivarium-process",
                    *swap_flags, "--out-root", out_c], cwd=REPO, check=True)


def _render(invest_ref: str, out: str, max_seeds: int, study: str | None = None) -> None:
    argv = [PY, "scripts/comparison_report_card.py",
            "--investigation", invest_ref, "--out", out,
            "--local-pbg-seeds", str(max(1, max_seeds))]
    if study:
        argv += ["--study", study]
    subprocess.run(argv, cwd=REPO, check=True)


def run_study(spec, out: str = "out/report", mode: str = "serial",
              render_only: bool = False) -> int:
    """Run one study: engines (unless render_only) -> render just this study ->
    write its verdict -> materialize its report_cards/behavior_tests."""
    if not render_only:
        _run_engines(spec, out, mode)
    _render(spec.invest_name, out, spec.seeds, study=spec.name)
    materialize_study(spec)
    print(f"study '{spec.name}' done -> {out}/{spec.name}; verdict + tests materialized")
    return 0


def run_investigation(inv_ref: str, out: str = "out/report", mode: str = "serial",
                      render_only: bool = False) -> int:
    """Run every study in an investigation, then render + verdict + materialize."""
    from scripts._compare.study_spec import _context, _invest_dir, specs_from_configs
    ctx = _context(_invest_dir(inv_ref))
    specs = specs_from_configs(ctx)
    if not specs:
        raise SystemExit(f"investigation {inv_ref!r} has no studies")
    if not render_only:
        for spec in specs:
            _run_engines(spec, out, mode)
    max_seeds = max((s.seeds for s in specs), default=1)
    _render(inv_ref, out, max_seeds)
    for spec in specs:
        materialize_study(spec)
    print(f"investigation '{inv_ref}' done: {len(specs)} studies -> {out}")
    return 0


# Re-exported for the CLI so it can resolve a study/investigation ref to a spec.
__all__ = ["run_study", "run_investigation", "load_study", "load_investigation"]
