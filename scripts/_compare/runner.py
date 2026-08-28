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

from scripts._compare.study_spec import (REPO, is_reference_config,
                                         load_investigation, load_study)
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
    is_path = is_reference_config(spec.config)
    swap_flags = ["--from-vecoli-config", spec.config] if is_path else []
    # Companion fork processes the study names (see StudySpec.inject_processes).
    # Passed on BOTH engine invocations: the genuine-vEcoli side ignores the flag
    # (its own config already lists the process), so a single list stays the
    # study's one declaration rather than two that can drift apart.
    for _p in getattr(spec, "inject_processes", None) or []:
        swap_flags += ["--inject-process", _p]
    # Metabolic exchange fluxes to emit onto listeners.exchange_flux.<leaf> on
    # BOTH arms (e.g. the violacein card's rate/yield inputs). Same flags on each
    # engine so candidate and reference emit the same leaves.
    flux_flags = [f
                  for leaf, key in (spec.exchange_fluxes or {}).items()
                  for f in ("--exchange-flux", f"{leaf}={key}")]
    # The BASIS rides with the map on BOTH invocations. Declared once by the
    # study so the two arms cannot report different quantities under one leaf
    # name; each engine reaches it by its own route (the candidate differences
    # its counts store, the reference reads the wrapped metabolism's own
    # gDCW-basis listener, which requires a metabolism that keys that leaf by
    # metabolite id).
    if flux_flags:
        flux_flags += ["--exchange-flux-basis",
                       str(getattr(spec, "exchange_flux_basis", None) or "counts")]
    # ⛔⛔ THE VARIANT RIDES THE REFERENCE INVOCATION ONLY, AND ONLY WHEN THE
    # STUDY DECLARED IT. Unlike the flux flags above, this is NOT symmetric: the
    # reference arm applies the config's variant through the fork's own
    # `apply_variant`, while the candidate arm takes its perturbation from
    # `--cache-dir`.
    # ⚠ CORRECTED: an earlier version of this comment said passing it to both
    # "would apply the same perturbation twice on the candidate side". That is
    # false and checkable — `make_run_one`'s `v2ecoli` branch never reads
    # `variant` at all. The flag would simply be inert there. The asymmetry is
    # still right, but for the plain reason and not a scary one.
    # ⛔ The real hazard on the candidate side is UNCHECKED and lives elsewhere:
    # this asymmetry is only correct if `v2_cache` is a perturbation-baked cache.
    # Nothing validates that, so a study pointing both arms at a stock cache
    # compares perturbed-vs-baseline and reads as a clean comparison.
    # ⚠ `is not None`, not truthiness: `variant: 0` is a study saying "baseline,
    # deliberately", and it must reach the runner as an explicit choice — passing
    # nothing instead would trip the runner's own refusal, which is exactly the
    # question the study already answered.
    variant_flags = ([] if getattr(spec, "variant", None) is None
                     else ["--variant", str(spec.variant)])
    # Arbitrary listener leaves ("group.leaf") to emit as measurements on BOTH
    # arms — the general observable-declaration hook.
    obs_flags = [f for o in (spec.observables or [])
                 for f in ("--observable", str(o))]
    # Bulk molecule KPIs (config-specific): emitted on BOTH arms under
    # listeners.observable_bulk.<id> for the bulk-aware comparison cards.
    obs_bulk_flags = [f for i in (spec.observable_bulk_ids or [])
                      for f in ("--observable-bulk", str(i))]
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "v2ecoli", "--condition", spec.condition,
                    "--cache-dir", spec.v2_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", v2_cap,
                    "--chunk", "60", "--mode", mode,
                    "--match-initial-state", "--match-vecoli-simdata", ref_sd,
                    *swap_flags, *flux_flags, *obs_flags, *obs_bulk_flags,
                    "--out-root", out_c], cwd=REPO, check=True)
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "vecoli", "--condition", spec.condition,
                    "--cache-dir", spec.ve_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", str(per_gen),
                    "--chunk", "60", "--mode", mode,
                    "--vecoli-source", "vivarium-process",
                    *swap_flags, *flux_flags, *variant_flags, *obs_flags,
                    *obs_bulk_flags,
                    "--out-root", out_c], cwd=REPO, check=True)


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
