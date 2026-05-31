"""Upgrade auto-generated embed names/descriptions on the canonical
closeout artifacts. Idempotent — only updates entries whose
description starts with the auto-wiring sentinel.
"""
from __future__ import annotations
from pathlib import Path
import yaml

# (study_slug, fig_url) → (display_name, description).
UPGRADES: dict[tuple[str, str], tuple[str, str]] = {
    # ---- pdmp-00 -------------------------------------------------
    ("pdmp-00-characterization",
     "/reports/figures/pdmp-00/per_condition_growth_rate.html"):
    ("Phase 0 — per-condition growth-rate baseline",
     "v2ecoli baseline ensemble growth rates per nutrient condition "
     "(M9-glucose, M9-acetate, M9-glucose+aa). Phase 0 reference for "
     "every subsequent study."),
    ("pdmp-00-characterization",
     "/reports/figures/pdmp-00/profile_decomposition_bars.html"):
    ("Phase 0 — per-process compute decomposition",
     "Per-tick cost decomposed by Process. Reference profile that "
     "Phase 4 sprint 1 replaced with the marshalling-aware version."),
    ("pdmp-00-characterization",
     "/reports/figures/pdmp-00/rng_seeding_fix_proof.html"):
    ("Phase 0 — RNG seeding fix verification",
     "Empirical proof that per-process RNG seeding correctly "
     "decorrelates seed→trajectory mapping."),
    ("pdmp-00-characterization",
     "/reports/figures/pdmp-00/variable_categorization_bars.html"):
    ("Phase 0 — Markov-blanket variable categorization",
     "Categorical breakdown of the WCM state into input / sufficient-"
     "statistic / output buckets — the interface contract every later "
     "study tests against."),

    # ---- pdmp-01 -------------------------------------------------
    ("pdmp-01-metabolism-ode",
     "/reports/figures/pdmp-01/kinetic_constraint_curves.html"):
    ("Phase 1 — kinetic-constraint curves",
     "Michaelis-Menten kinetic constraint envelopes per substrate."),
    ("pdmp-01-metabolism-ode",
     "/reports/figures/pdmp-01/pdmp_vs_baseline.html"):
    ("Phase 1 — PDMP metabolism vs v2ecoli baseline",
     "Trajectory overlay: MillardPDMPMetabolism + consumption_matched "
     "ref-growth driver vs the Phase-0 reference ensemble on "
     "M9-glucose. PR #72 closeout artifact."),
    ("pdmp-01-metabolism-ode",
     "/reports/figures/pdmp-01/pdmp_vs_phase0.html"):
    ("Phase 1 — PDMP vs Phase-0 ensemble (W₂)",
     "W₂ distance between PDMP and Phase-0 ensembles on cell_mass + "
     "dry_mass. Within ±σ at t=600 s on M9-glucose."),
    ("pdmp-01-metabolism-ode",
     "/reports/figures/pdmp-01/phase1_progress.html"):
    ("Phase 1 — PR progress report",
     "Closeout summary for PR #72. Charts: bridge round-trip residual, "
     "FBA-bridge architecture, Millard kinetic ATP drain, W₂ gap "
     "vs Phase 0, per-condition growth rate."),

    # ---- pdmp-02 -------------------------------------------------
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/trajectory_divergence.html"):
    ("Phase 2 sprint 4 — trajectory-shape divergence",
     "|Δcell_mass| peaks at 150 fg at the rnap_data listener at "
     "t=600 s. Closeout finding: cell_mass is the WRONG observable for "
     "jump-process variance — consumption_matched homeostat washes "
     "per-tick variance out by construction."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/trajectory_divergence_poisson.html"):
    ("Phase 2 — trajectory divergence in `poisson` mode",
     "Same as the aggregate trajectory divergence, restricted to the "
     "per-promoter / per-protein Poisson tau-leap branch."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/trajectory_divergence_discrete.html"):
    ("Phase 2 — trajectory divergence in `discrete` mode",
     "Same as the aggregate trajectory divergence, restricted to the "
     "original discrete-time stochastic branch — for side-by-side "
     "comparison with the Poisson mode."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/ensemble_validation_period60.html"):
    ("Phase 2 sprint 9 — sparse-injection ensemble validation",
     "60-tick injection period: homeostat washes per-tick variance, "
     "confirming the closeout finding."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/ensemble_validation_tau60.html"):
    ("Phase 2 sprint 9 — τ=60 ensemble validation",
     "Same sparse-injection experiment with τ=60 averaging window."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/initiation_modes_comparison.html"):
    ("Phase 2 sprint 2 — discrete vs Poisson trajectory overlay",
     "Side-by-side trajectories under the two initiation modes for "
     "TranscriptInitiation and PolypeptideInitiation."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/division_time_distribution.html"):
    ("Phase 2 — division time distribution",
     "First-passage division-time distribution from the stochastic "
     "bisection driver."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/inheritance_distribution_v2.html"):
    ("Phase 2 — inheritance distribution",
     "Daughter1 − daughter2 mass-split distribution at division — "
     "validates the partitioning mechanism."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/phase2_event_rates.html"):
    ("Phase 2 — per-cell event rates across processes",
     "Per-cell event-rate magnitudes spanning 6+ orders of magnitude "
     "— motivates the hybrid exact-SSA / τ-leaping topology."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/gillespie_trajectory.html"):
    ("Phase 2 — Gillespie single-cell trajectory",
     "Single-cell exact-SSA mRNA/protein trajectory diagnostic."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/stochastic_gene_trajectories.html"):
    ("Phase 2 — stochastic gene trajectories",
     "Per-gene RNA-init + ribosome-init event-stream diagnostic."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/waiting_time_distribution.html"):
    ("Phase 2 — inter-event waiting-time distribution",
     "Inter-event waiting times — diagnostic for exponentiality under "
     "the Poisson tau-leap mode."),
    ("pdmp-02-jump-processes",
     "/reports/figures/pdmp-02/ensemble_validation.html"):
    ("Phase 2 — initial ensemble validation",
     "Phase-2 baseline ensemble vs Phase-0 reference before "
     "homeostat tuning landed."),

    # ---- pdmp-03 -------------------------------------------------
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_count_metric.html"):
    ("Phase 3 sprint 12 — count-based ABC distance",
     "Combined count vector sqrt(d_rna² + d_ribosome²) reaches 3.06× "
     "truth-vs-next-nearest separation — the Phase-3-canonical ABC "
     "distance, 100× sharper Ts identification than aggregate "
     "log-likelihood."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_multi_observable.html"):
    ("Phase 3 sprint 11 — multi-observable ABC",
     "Per-channel vs aggregate ABC. Diagnoses the self-calibration "
     "issue that motivated the pivot to count-based distances."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_smc_2d.html"):
    ("Phase 3 sprint 10 — 2D (Ts, Ps) ABC sweep",
     "Reveals the anti-diagonal ridge under aggregate log-likelihood: "
     "reducing transcription while raising translation partially "
     "cancels. Per-channel distances break the ridge."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_smc_sequential.html"):
    ("Phase 3 sprint 9 — sequential ε refinement",
     "Posterior collapses to {truth} as ε tightens through "
     "p95/p75/p50/p25/p05 quantiles."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_smc_stub.html"):
    ("Phase 3 sprint 7 — ABC-SMC stub",
     "First ABC stub at scale-vs-distance, before the noise-floor "
     "recalibration."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/abc_posterior_shrinkage.html"):
    ("Phase 3 — ABC-SMC posterior shrinkage over rounds",
     "Posterior shrinkage diagnostic over SMC generations."),
    ("pdmp-03-inference",
     "/reports/figures/pdmp-03/likelihood_ensemble.html"):
    ("Phase 3 sprint 5 — ensemble likelihood figure",
     "Per-replicate log-likelihood trajectories + intra-ensemble "
     "noise floor band."),

    # ---- pdmp-04 -------------------------------------------------
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/per_step_compute_decomposition.html"):
    ("Phase 4 sprint 1 — per-tick compute decomposition",
     "76% framework overhead, no single Process dominates. Motivates "
     "the column-centric runtime as the architectural answer."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/marshalling_hotspots.html"):
    ("Phase 4 sprint 2 — function-level marshalling hotspots",
     "13,817 isinstance + 4,213 __array_finalize__ + 52 process_update "
     "calls per tick. Marshalling dominates over per-Process compute."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/column_centric_prototype.html"):
    ("Phase 4 sprint 3 — column-centric runtime prototype",
     "Constant per-trajectory wall regardless of N — 3.1× speedup on "
     "pure-Poisson toy."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/multistep_prototype.html"):
    ("Phase 4 sprint 4 — 3-step composite prototype",
     "Speedup does NOT compound across processes — pbg overhead is "
     "per-process, not per-step."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/wcm_baseline_projection.html"):
    ("Phase 4 sprint 5 — real WCM baseline + 1500× projection",
     "Real WCM benchmark: pbg sequential at N=10³ × 30 ticks = 26 min; "
     "column-centric parallel projection ≈ 1 sec = ~1500× total "
     "speedup. The Phase-4 closeout headline."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/ti_shaped_prototype.html"):
    ("Phase 4 sprint 6 — TI-shaped column-centric runner",
     "Realistic TI-shaped workload: honest 1.5× per-tick recalibration "
     "(vs toys' 3×). Final number behind the ~1500× projection."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/jax_julia_speedup.html"):
    ("Phase 4 — JAX/Julia backend speedup (Phase 1 wrap-up)",
     "JAX vs basico backend speedup for the Millard ODE on the "
     "standalone benchmark (Phase 1 perf #4 closeout)."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/jit_compile_cost.html"):
    ("Phase 4 — JIT compile-cost amortization",
     "JAX compile-time vs run-time crossover as a function of "
     "trajectory count and trajectory length."),
    ("pdmp-04-compilation",
     "/reports/figures/pdmp-04/memory_footprint.html"):
    ("Phase 4 — per-cell memory footprint by bucket",
     "Memory budget per WCM cell broken down by listener / state / "
     "ParCa cache."),

    # ---- pdmp-05 -------------------------------------------------
    ("pdmp-05-causal-discovery",
     "/reports/figures/pdmp-05/active_inference_eig.html"):
    ("Phase 5 — active-inference EIG (planned)",
     "Expected information gain for in-silico interventions under "
     "candidate gene-function hypotheses. Planned for sprint 5+."),
    ("pdmp-05-causal-discovery",
     "/reports/figures/pdmp-05/intervention_design_eig.html"):
    ("Phase 5 — intervention design EIG (planned)",
     "Companion to active-inference EIG: per-intervention EIG ranking. "
     "Planned for sprint 5+."),
    ("pdmp-05-causal-discovery",
     "/reports/figures/pdmp-05/pc_algorithm_dag.html"):
    ("Phase 5 — PC algorithm DAG (planned)",
     "Recovered causal DAG via PC algorithm on Phase-3 likelihood "
     "summaries. Planned for sprint 5+."),
    ("pdmp-05-causal-discovery",
     "/reports/figures/pdmp-05/bayes_factor_per_gene.html"):
    ("Phase 5 — per-gene Bayes factor (planned)",
     "Per-gene Bayes factor against null annotation under the "
     "Phase-5 marginal-likelihood pipeline. Planned for sprint 4+."),
    ("pdmp-05-causal-discovery",
     "/reports/figures/pdmp-05/bayes_factor_per_gene_v2.html"):
    ("Phase 5 — per-gene Bayes factor (v2 stub)",
     "Earlier hand-coded stub. Superseded by sprint-1 (bayes_factor_"
     "stub.html) and sprint-2 (pseudo_marginal_diagnostic.html)."),
}


def main():
    sentinel = "Auto-wired existing figure at"
    n_upgraded = 0
    for sy in sorted(Path("studies").glob("pdmp-*/study.yaml")):
        slug = sy.parent.name
        data = yaml.safe_load(sy.read_text())
        changed = False
        for entry in data.get("embed_visualizations", []) or []:
            key = (slug, entry["url"])
            if key in UPGRADES:
                new_name, new_desc = UPGRADES[key]
                entry["name"] = new_name
                entry["description"] = new_desc
                changed = True
                n_upgraded += 1
        if not changed:
            continue
        text = sy.read_text()
        lines = text.splitlines()
        cut = next(
            (i for i, ln in enumerate(lines)
             if ln.startswith("embed_visualizations:")), None)
        head = "\n".join(lines[:cut]) if cut is not None else ""
        tail = yaml.dump(
            {"embed_visualizations": data["embed_visualizations"]},
            default_flow_style=False, allow_unicode=True,
            sort_keys=False, width=120,
        )
        sy.write_text(head.rstrip() + "\n" + tail)
    print(f"Upgraded {n_upgraded} embed entries across studies.")


if __name__ == "__main__":
    main()
