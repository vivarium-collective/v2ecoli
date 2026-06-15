# Phase-3 inference (increment 1): SBC-validated ABC-SMC on a WCM-calibrated count surrogate — design

**Date:** 2026-06-12
**Study:** pdmp-03-inference (v2ecoli-pdmp investigation)
**Status:** design approved; ready for implementation plan

## Goal

Deliver a *real* Bayesian inference result for Phase 3: an ABC-SMC posterior
recovery of a transcription-initiation rate from count-level observations,
validated by Simulation-Based Calibration (SBC) and a posterior-predictive
check (PPC). This is the genuine pdmp-03 gate
(`abc-posterior-recovery-on-synthetic` + `sbc-calibration-uniform-rank`), and
it must run on a laptop.

This replaces the existing `scripts/phase3_abc_*.py` grid-separation
diagnostics (real outputs, but the prior audit established they are toy 3x3
grid separations on a self-generated grid — NOT a posterior, credible interval,
or calibrated ABC) with a proper ABC-SMC + SBC pipeline. The cited-but-missing
`run_sbc.py` / `run_abc_smc.py` / `run_ppc.py` are built here.

## Decisions locked (via brainstorming)

- **Forward model:** a fast count-likelihood surrogate, NOT the full WCM in the
  ABC loop. The WCM is run ONCE to calibrate the surrogate, then the ABC-SMC and
  SBC loops draw from the cheap surrogate. (Full-WCM coupling is infeasible —
  ABC-SMC+SBC would be weeks/months — and is deferred to the Phase-4 compiled
  runtime.)
- **Inference target:** a SINGLE parameter `theta` = transcription-init rate
  scale (the channel pdmp-02 validated as carrying the jump-process variance via
  `rnap_data.rna_init_event`).
- **Likelihood:** negative-binomial on counts (the count modality pdmp-03's
  hypothesis specifies), calibrated to real WCM statistics.

## Why this is the right observable

pdmp-02 confirmed (~30x cross-seed CV) that jump-process variance lives in the
per-event count `rna_init_event`, not in aggregate mass/pool observables. So
inferring a transcription rate from count observations is the principled Phase-3
deliverable, and the negative-binomial count likelihood is the matching
observation model.

## Components (each independently testable)

### 1. Surrogate calibration — `v2ecoli/inference/count_surrogate.py`
`calibrate_count_surrogate(n_seeds, n_steps) -> {mu0, phi, theta_ref}`:
run the real WCM baseline at the reference transcription scale, measure the
per-tick `rnap_data.rna_init_event` total across seeds/ticks, and fit a
negative-binomial: mean `mu0`, overdispersion `phi` (via method-of-moments or
MLE on the empirical mean+variance). `theta_ref = 1.0` is the reference scale.
Persist the calibration to `.pbg/runs/phase3-surrogate-calibration.json`. The
WCM run reuses the pdmp-02 measurement path (scripts/phase2_count_variance.py).

Interface: pure given the measured stats; the WCM measurement is a separate
step so calibration is unit-testable with injected stats.

### 2. Forward model — `v2ecoli/inference/count_surrogate.py`
`count_surrogate_sample(theta, calib, n_obs, rng) -> np.ndarray`:
draw `n_obs` synthetic count observations from
`NegBinom(mean = theta * calib.mu0, dispersion = calib.phi)`. Vectorized,
deterministic given `rng`. This is the cheap forward model the ABC/SBC loops
call thousands of times.

`count_summary(counts) -> np.ndarray`: the summary statistics the ABC distance
uses (e.g. [mean, variance, or quantiles]) — a small fixed vector.

### 3. ABC-SMC — `scripts/run_abc_smc.py` + `v2ecoli/inference/abc_smc.py`
`abc_smc(observed_summary, calib, prior, *, n_particles, eps_schedule, rng)
-> {particles, weights, eps_trace, ess_trace, accept_trace}`:
a proper sequential Monte Carlo over `theta`:
- particle population of size `n_particles`;
- a DECREASING tolerance schedule `eps_schedule` (e.g. adaptive quantile of the
  previous generation's distances);
- importance WEIGHTS updated each generation (prior x kernel ratio);
- a perturbation KERNEL (Gaussian, bandwidth from the weighted particle
  variance) between generations;
- distance = Euclidean on `count_summary(sim) vs observed_summary`.
Reports posterior ESS and acceptance rate per generation (the diagnostics the
study's uq_standards require). This is the SMC the audit found missing (the
existing pilot is iterated rejection-ABC with box-narrowing, no weights/kernel).

`scripts/run_abc_smc.py`: CLI that calibrates (or loads calibration), generates
ONE synthetic observed dataset at a fixed `theta_true`, runs `abc_smc`, and
writes the posterior + a figure
`reports/figures/pdmp-03/abc_smc_posterior.html` showing the posterior
concentrating on `theta_true` (the headline recovery).

### 4. SBC — `scripts/run_sbc.py` + `v2ecoli/inference/sbc.py`
`run_sbc(calib, prior, *, n_sbc, abc_kwargs, rng) -> {ranks, chi2_p, hist}`:
for `n_sbc` (>= 150) draws: sample `theta* ~ prior`, simulate observed counts,
run `abc_smc`, compute the rank of `theta*` within the weighted posterior
sample. The rank histogram must be uniform — `chi2_p` from a chi-square
goodness-of-fit; PASS if `chi2_p > 0.05`. Writes
`reports/figures/pdmp-03/sbc_rank_histogram.html`. This is the load-bearing
calibration gate.

### 5. PPC — `scripts/run_ppc.py`
Posterior-predictive coverage: draw `theta` from the posterior, simulate counts,
form posterior-predictive 90% intervals on the summary statistics, and report
the empirical coverage of held-out observed statistics (target >= ~85-90%).
Writes `reports/figures/pdmp-03/ppc_coverage.html`.

## Data flow
1. WCM (once) -> calibrate surrogate -> {mu0, phi}.
2. Recovery: fix theta_true -> simulate observed -> abc_smc -> posterior near
   theta_true (figure + summary).
3. SBC: K x [theta*~prior -> simulate -> abc_smc -> rank] -> rank histogram +
   chi2 uniformity (PASS/FAIL).
4. PPC: posterior -> predictive intervals -> coverage of observed statistics.

## Gates delivered (pdmp-03 acceptance)
- `abc-posterior-recovery-on-synthetic`: the 90% credible interval contains
  `theta_true` and its width shrinks as `n_obs` grows (report both).
- `sbc-calibration-uniform-rank`: rank-histogram chi2 p > 0.05 over n_sbc >= 150.
- `ppc-coverage`: posterior-predictive 90% intervals cover >= ~85% of held-out
  summary statistics.

## Scope (YAGNI)
- ONE parameter (`theta` = transcription-init scale). Multi-parameter (Ts,Ps)
  and identifiability/FIM are a later increment.
- The surrogate IS the likelihood; the WCM is run once for calibration, never in
  the ABC/SBC loops.
- Negative-binomial count likelihood only (the spec's count modality).
- Full-WCM-in-the-loop inference deferred to Phase 4 (compiled runtime).

## Testing
- `count_surrogate`: calibration recovers known (mu0, phi) from injected stats;
  `count_surrogate_sample` mean/variance match the NegBinom moments within MC
  error at large `n_obs`.
- `abc_smc`: recovers a known mean on a trivial Gaussian forward model (the SMC
  machinery is correct independent of the surrogate); ESS/weights are finite and
  the posterior concentrates as eps decreases.
- `sbc`: on a self-consistent toy (forward == inference model) the rank
  histogram is uniform (chi2 p > 0.05) at n_sbc>=150 — the correct-by-
  construction check that the SBC implementation is right.
- Determinism: all stochastic steps take an explicit `rng` (np.random.Generator)
  so tests are reproducible.

## Risks
- **SBC compute:** n_sbc x n_particles x generations surrogate draws. With the
  cheap surrogate this is seconds-to-minutes; if n_sbc=150 is slow, the surrogate
  (not the WCM) is the only cost and is trivially parallelizable.
- **Surrogate fidelity:** the NegBinom may not capture all WCM count structure
  (temporal correlation, per-cistron heterogeneity). v1 uses the aggregate
  per-tick total; richer count structure is a later increment. The honest claim
  is "ABC-SMC+SBC validated on a WCM-calibrated count likelihood," not "on the
  full WCM."
- **theta identifiability:** a single scale on a NegBinom mean is identifiable by
  construction; SBC will catch it if not.
