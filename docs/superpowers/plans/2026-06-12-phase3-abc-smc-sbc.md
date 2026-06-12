# Phase-3 ABC-SMC + SBC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a transcription-init rate (theta) from count-level observations via a real ABC-SMC, validated by SBC (rank-histogram uniformity) + PPC, using a negative-binomial count surrogate calibrated once to real WCM `rna_init_event` statistics.

**Architecture:** Three pure, WCM-free units (`count_surrogate`, `abc_smc`, `sbc`) form the inference core and are fully unit-testable. One isolated task runs the WCM once to calibrate the surrogate. Three CLIs (`run_abc_smc`/`run_sbc`/`run_ppc`) wire them into figures + the pdmp-03 gates.

**Tech Stack:** Python, numpy (`np.random.Generator`), scipy.stats (chi-square), matplotlib (figures). No WCM in the ABC/SBC loops.

**Reference spec:** `docs/superpowers/specs/2026-06-12-phase3-abc-smc-sbc-design.md`

**Run tests with `.venv/bin/python -m pytest` (bare python lacks `unum`).** All stochastic functions take an explicit `rng = np.random.default_rng(seed)` for reproducibility.

---

## Task 1: count surrogate (calibration + forward model + summary)

**Files:**
- Create: `v2ecoli/inference/__init__.py` (empty)
- Create: `v2ecoli/inference/count_surrogate.py`
- Test: `tests/test_count_surrogate.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_count_surrogate.py
import numpy as np
from v2ecoli.inference.count_surrogate import (
    fit_negbinom_dispersion, count_surrogate_sample, count_summary)

def test_fit_dispersion_method_of_moments():
    # var = mean + mean^2/phi  =>  phi = mean^2 / (var - mean)
    assert abs(fit_negbinom_dispersion(mean=10.0, var=30.0) - 5.0) < 1e-9
    # var <= mean -> Poisson limit (inf dispersion)
    assert not np.isfinite(fit_negbinom_dispersion(mean=10.0, var=10.0))

def test_sample_moments_match_calibration():
    rng = np.random.default_rng(0)
    calib = {"mu0": 40.0, "phi": 8.0}
    x = count_surrogate_sample(theta=1.0, calib=calib, n_obs=200000, rng=rng)
    # mean ~ theta*mu0 = 40 ; var ~ mean + mean^2/phi = 40 + 1600/8 = 240
    assert abs(x.mean() - 40.0) < 1.0
    assert abs(x.var() - 240.0) < 12.0

def test_theta_scales_mean():
    rng = np.random.default_rng(1)
    calib = {"mu0": 40.0, "phi": 8.0}
    x = count_surrogate_sample(theta=1.5, calib=calib, n_obs=200000, rng=rng)
    assert abs(x.mean() - 60.0) < 1.5

def test_summary_shape():
    s = count_summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s.shape == (2,)            # [mean, std]
    assert abs(s[0] - 2.5) < 1e-9
```

- [ ] **Step 2: Run -> FAIL.** `.venv/bin/python -m pytest tests/test_count_surrogate.py -q`

- [ ] **Step 3: Implement**
```python
# v2ecoli/inference/count_surrogate.py
from __future__ import annotations
import numpy as np

def fit_negbinom_dispersion(mean: float, var: float) -> float:
    """Method-of-moments dispersion phi for NegBinom (var = mean + mean^2/phi).
    Returns +inf when var <= mean (Poisson limit)."""
    if var <= mean:
        return float("inf")
    return float(mean * mean / (var - mean))

def count_surrogate_sample(theta, calib, n_obs, rng):
    """Draw n_obs counts from NegBinom(mean=theta*mu0, dispersion=phi)."""
    mean = float(theta) * float(calib["mu0"])
    phi = float(calib["phi"])
    if not np.isfinite(phi):
        return rng.poisson(mean, size=n_obs).astype(float)
    p = phi / (phi + mean)           # numpy negative_binomial(n=phi, p)
    return rng.negative_binomial(phi, p, size=n_obs).astype(float)

def count_summary(counts) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    return np.array([c.mean(), c.std()])
```

- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/__init__.py v2ecoli/inference/count_surrogate.py tests/test_count_surrogate.py
git commit -m "feat(pdmp-03): negative-binomial count surrogate (calibration + forward model)"
```

---

## Task 2: ABC-SMC engine

**Files:**
- Create: `v2ecoli/inference/abc_smc.py`
- Test: `tests/test_abc_smc.py`

- [ ] **Step 1: Write the failing test** (recover a known mean on a Gaussian forward model — the SMC machinery is correct independent of the surrogate)
```python
# tests/test_abc_smc.py
import numpy as np
from v2ecoli.inference.abc_smc import abc_smc

def test_abc_smc_recovers_gaussian_mean():
    rng = np.random.default_rng(0)
    theta_true = 3.0
    # forward model: summary = sample mean of 50 Normal(theta, 1) draws
    def simulate(theta, r):
        return np.array([r.normal(theta, 1.0, size=50).mean()])
    observed = simulate(theta_true, np.random.default_rng(99))
    res = abc_smc(observed, simulate, prior=(0.0, 6.0),
                  n_particles=400, n_generations=5, rng=rng)
    post_mean = np.average(res["particles"], weights=res["weights"])
    assert abs(post_mean - theta_true) < 0.3            # recovers truth
    assert res["eps_trace"][-1] < res["eps_trace"][0]   # eps decreased
    assert np.all(np.isfinite(res["weights"])) and abs(res["weights"].sum() - 1.0) < 1e-9

def test_abc_smc_posterior_concentrates():
    rng = np.random.default_rng(1)
    def simulate(theta, r):
        return np.array([r.normal(theta, 1.0, size=50).mean()])
    observed = np.array([3.0])
    res = abc_smc(observed, simulate, prior=(0.0, 6.0),
                  n_particles=400, n_generations=6, rng=rng)
    # posterior std shrinks well below the prior std (uniform[0,6] std ~1.73)
    var = np.average((res["particles"] - np.average(res["particles"], weights=res["weights"]))**2,
                     weights=res["weights"])
    assert var**0.5 < 0.5
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement** (Toni et al. 2009 ABC-SMC: weighted population, perturbation kernel, adaptive eps from the previous generation's distance quantile)
```python
# v2ecoli/inference/abc_smc.py
from __future__ import annotations
import numpy as np

def abc_smc(observed_summary, simulate, prior, *, n_particles=400,
            n_generations=5, eps_quantile=0.5, rng):
    """Sequential Monte Carlo ABC over a scalar theta.

    simulate(theta, rng) -> summary vector (same shape as observed_summary).
    prior = (lo, hi) uniform. Returns particles, normalized weights, and
    eps/ess/accept traces. Distance = Euclidean on summaries.
    """
    lo, hi = float(prior[0]), float(prior[1])
    obs = np.asarray(observed_summary, dtype=float)

    def dist(theta, r):
        return float(np.linalg.norm(np.asarray(simulate(theta, r), float) - obs))

    # Generation 0: sample from prior, keep all, record distances.
    parts = rng.uniform(lo, hi, size=n_particles)
    dists = np.array([dist(t, rng) for t in parts])
    weights = np.full(n_particles, 1.0 / n_particles)
    eps_trace, ess_trace, accept_trace = [float(dists.max())], [], []

    for _ in range(1, n_generations):
        eps = float(np.quantile(dists, eps_quantile))   # tighten tolerance
        # kernel bandwidth = weighted particle std (Gaussian perturbation)
        mean_w = np.average(parts, weights=weights)
        var_w = np.average((parts - mean_w) ** 2, weights=weights)
        kstd = max(np.sqrt(2.0 * var_w), 1e-6)
        new_parts = np.empty(n_particles)
        new_w = np.empty(n_particles)
        new_d = np.empty(n_particles)
        tries = 0
        i = 0
        while i < n_particles:
            tries += 1
            j = rng.choice(n_particles, p=weights)       # pick from prev pop
            cand = parts[j] + rng.normal(0.0, kstd)       # perturb
            if cand < lo or cand > hi:
                continue
            d = dist(cand, rng)
            if d <= eps:
                # importance weight: prior / sum_k w_k K(cand|theta_k)
                kern = np.exp(-0.5 * ((cand - parts) / kstd) ** 2)
                denom = np.sum(weights * kern)
                new_parts[i] = cand
                new_w[i] = (1.0 / (hi - lo)) / denom if denom > 0 else 0.0
                new_d[i] = d
                i += 1
        parts, dists = new_parts, new_d
        s = new_w.sum()
        weights = new_w / s if s > 0 else np.full(n_particles, 1.0 / n_particles)
        eps_trace.append(eps)
        ess_trace.append(float(1.0 / np.sum(weights ** 2)))
        accept_trace.append(float(n_particles / max(tries, 1)))

    return {"particles": parts, "weights": weights, "eps_trace": eps_trace,
            "ess_trace": ess_trace, "accept_trace": accept_trace}
```

- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/abc_smc.py tests/test_abc_smc.py
git commit -m "feat(pdmp-03): real ABC-SMC (weights + perturbation kernel + adaptive eps)"
```

---

## Task 3: SBC (rank computation + uniformity)

**Files:**
- Create: `v2ecoli/inference/sbc.py`
- Test: `tests/test_sbc.py`

- [ ] **Step 1: Write the failing test** (correct-by-construction: forward == inference model -> ranks uniform)
```python
# tests/test_sbc.py
import numpy as np
from v2ecoli.inference.sbc import posterior_rank, run_sbc

def test_posterior_rank_basic():
    # theta_star at the median of the posterior sample -> mid rank
    post = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = np.full(5, 0.2)
    assert posterior_rank(2.0, post, w) == 2     # 2 particles below 2.0

def test_sbc_ranks_uniform_on_self_consistent_toy():
    rng = np.random.default_rng(0)
    # forward + inference share the same Gaussian model -> SBC must be uniform
    def simulate(theta, r):
        return np.array([r.normal(theta, 1.0, size=30).mean()])
    res = run_sbc(simulate, prior=(0.0, 6.0), n_sbc=150,
                  abc_kwargs=dict(n_particles=200, n_generations=4),
                  rng=rng)
    assert res["chi2_p"] > 0.05                  # rank histogram is uniform
    assert len(res["ranks"]) == 150
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement**
```python
# v2ecoli/inference/sbc.py
from __future__ import annotations
import numpy as np
from scipy import stats
from v2ecoli.inference.abc_smc import abc_smc

def posterior_rank(theta_star, particles, weights) -> int:
    """Rank of theta_star = number of posterior particles strictly below it.
    (Weight-agnostic count rank; SBC theory holds for the empirical sample.)"""
    return int(np.sum(np.asarray(particles) < float(theta_star)))

def run_sbc(simulate, prior, *, n_sbc=150, abc_kwargs, rng):
    """For n_sbc draws: theta* ~ prior, simulate observed, ABC-SMC posterior,
    record rank(theta*). Return ranks, chi2 p-value of rank-uniformity, hist."""
    lo, hi = float(prior[0]), float(prior[1])
    ranks = []
    for _ in range(n_sbc):
        theta_star = rng.uniform(lo, hi)
        observed = simulate(theta_star, rng)
        res = abc_smc(observed, simulate, prior=prior, rng=rng, **abc_kwargs)
        ranks.append(posterior_rank(theta_star, res["particles"], res["weights"]))
    ranks = np.array(ranks)
    n_part = abc_kwargs["n_particles"]
    n_bins = 20
    hist, edges = np.histogram(ranks, bins=n_bins, range=(0, n_part))
    expected = len(ranks) / n_bins
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    chi2_p = float(stats.chi2.sf(chi2, df=n_bins - 1))
    return {"ranks": ranks, "chi2_p": chi2_p, "hist": hist, "edges": edges}
```

- [ ] **Step 4: Run -> PASS** (may take ~30-60s for 150 ABC runs; acceptable).
- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/sbc.py tests/test_sbc.py
git commit -m "feat(pdmp-03): SBC rank computation + chi-square uniformity gate"
```

---

## Task 4: WCM surrogate calibration (the single WCM run)

**Files:**
- Create: `scripts/calibrate_count_surrogate.py`
- Modify: `v2ecoli/inference/count_surrogate.py` (add `calibrate_from_counts`)
- Test: `tests/test_count_surrogate.py` (add a test for `calibrate_from_counts`)

- [ ] **Step 1: Add failing test for the pure calibration reducer**
```python
# append to tests/test_count_surrogate.py
def test_calibrate_from_counts():
    import numpy as np
    from v2ecoli.inference.count_surrogate import calibrate_from_counts
    rng = np.random.default_rng(0)
    # synthetic per-seed cumulative counts with known mean/var
    samples = rng.negative_binomial(8, 8/(8+40), size=5000).astype(float)
    calib = calibrate_from_counts(samples)
    assert abs(calib["mu0"] - 40) < 2
    assert abs(calib["phi"] - 8) < 3
    assert calib["theta_ref"] == 1.0
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement the reducer + the WCM script**
```python
# append to v2ecoli/inference/count_surrogate.py
def calibrate_from_counts(samples) -> dict:
    """Fit the surrogate (mu0, phi) from measured per-replicate count samples."""
    import numpy as np
    s = np.asarray(samples, dtype=float)
    mean, var = float(s.mean()), float(s.var())
    return {"mu0": mean, "phi": fit_negbinom_dispersion(mean, var),
            "theta_ref": 1.0}
```
```python
# scripts/calibrate_count_surrogate.py
"""Run the WCM ONCE at the reference transcription scale; measure per-seed
cumulative rna_init_event; fit the count surrogate; persist calibration JSON.

Usage: .venv/bin/python scripts/calibrate_count_surrogate.py [--n-seeds 8] [--n-steps 250]
Writes .pbg/runs/phase3-surrogate-calibration.json
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
from v2ecoli import build_composite
from v2ecoli.inference.count_surrogate import calibrate_from_counts

def _total(x):
    try: return float(np.nansum(np.asarray(x, dtype=float)))
    except Exception: return 0.0

def main(n_seeds, n_steps):
    cum = []
    for seed in range(n_seeds):
        c = build_composite("baseline", cache_dir="out/cache", seed=seed)
        ci = 0.0
        for _ in range(n_steps):
            c.run(1)
            L = (((c.state.get("agents") or {}).get("0") or {}).get("listeners") or {})
            rie = (L.get("rnap_data") or {}).get("rna_init_event")
            if rie is not None: ci += _total(rie)
        cum.append(ci); print(f"seed {seed}: cum_rna_init={ci:.0f}", flush=True)
    calib = calibrate_from_counts(np.array(cum))
    out = Path(".pbg/runs/phase3-surrogate-calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**calib, "n_seeds": n_seeds, "n_steps": n_steps,
                               "cum_rna_init": cum}, indent=2))
    print("calibration:", calib, "->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=250)
    a = ap.parse_args(); main(a.n_seeds, a.n_steps)
```

- [ ] **Step 4: Run the unit test -> PASS.** Then run the calibration once:
`.venv/bin/python scripts/calibrate_count_surrogate.py --n-seeds 8 --n-steps 250`
Expected: writes `.pbg/runs/phase3-surrogate-calibration.json` with mu0/phi > 0.
(NOTE: cross-seed cumulative counts may have low dispersion; if `var <= mean`,
`phi` is inf and the surrogate is Poisson — that's a valid, honest outcome to
record, not a failure.)

- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/count_surrogate.py scripts/calibrate_count_surrogate.py tests/test_count_surrogate.py
git commit -m "feat(pdmp-03): calibrate count surrogate from one WCM rna_init_event run"
```

---

## Task 5: run_abc_smc.py — headline posterior recovery + figure

**Files:**
- Create: `scripts/run_abc_smc.py`
- Test: `tests/test_run_abc_smc.py`

- [ ] **Step 1: Write the failing test** (recovery on the calibrated surrogate)
```python
# tests/test_run_abc_smc.py
import numpy as np
from scripts_phase3 import recover  # thin importable core (see step 3)

def test_recovery_ci_contains_truth():
    calib = {"mu0": 40.0, "phi": 8.0}
    res = recover(calib, theta_true=1.2, n_obs=400,
                  rng=np.random.default_rng(0),
                  n_particles=400, n_generations=6)
    lo, hi = res["ci90"]
    assert lo <= 1.2 <= hi                       # 90% CI contains truth
    assert (hi - lo) < 0.6                        # informative posterior
```
(The test imports the inference CORE; keep the CLI thin. Put the core in
`v2ecoli/inference/recover.py` and re-export, so it's importable without argparse.)

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement the core + CLI**
```python
# v2ecoli/inference/recover.py
from __future__ import annotations
import numpy as np
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary
from v2ecoli.inference.abc_smc import abc_smc

def recover(calib, theta_true, n_obs, rng, *, prior=(0.2, 3.0),
            n_particles=400, n_generations=6):
    def simulate(theta, r):
        return count_summary(count_surrogate_sample(theta, calib, n_obs, r))
    observed = simulate(theta_true, np.random.default_rng(12345))
    res = abc_smc(observed, simulate, prior=prior, rng=rng,
                  n_particles=n_particles, n_generations=n_generations)
    p, w = res["particles"], res["weights"]
    order = np.argsort(p); cw = np.cumsum(w[order])
    lo = float(p[order][np.searchsorted(cw, 0.05)])
    hi = float(p[order][np.searchsorted(cw, 0.95)])
    return {**res, "theta_true": theta_true, "post_mean": float(np.average(p, weights=w)),
            "ci90": (lo, hi)}
```
```python
# scripts/run_abc_smc.py  (thin CLI: load calibration, recover, write figure)
# also create scripts_phase3.py at repo root re-exporting recover for the test,
# OR import from v2ecoli.inference.recover in the test instead. Use the latter:
#   from v2ecoli.inference.recover import recover
# (update the test import accordingly).
```
NOTE: simplify the test import to `from v2ecoli.inference.recover import recover`
(drop the `scripts_phase3` shim). The CLI loads
`.pbg/runs/phase3-surrogate-calibration.json`, calls `recover`, and writes
`reports/figures/pdmp-03/abc_smc_posterior.html` (weighted posterior histogram
with theta_true + 90% CI marked).

- [ ] **Step 4: Run -> PASS.** Then run the CLI to produce the figure.
- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/recover.py scripts/run_abc_smc.py tests/test_run_abc_smc.py reports/figures/pdmp-03/abc_smc_posterior.html
git commit -m "feat(pdmp-03): run_abc_smc posterior recovery on the calibrated surrogate"
```

---

## Task 6: run_sbc.py — the SBC gate + figure

**Files:**
- Create: `scripts/run_sbc.py`
- Test: covered by `tests/test_sbc.py` (Task 3); add a calibrated-surrogate smoke test

- [ ] **Step 1: Add a calibrated-surrogate SBC test** (smaller n_sbc for speed)
```python
# append to tests/test_sbc.py
def test_sbc_on_calibrated_surrogate_runs_and_is_uniform():
    import numpy as np
    from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary
    from v2ecoli.inference.sbc import run_sbc
    calib = {"mu0": 40.0, "phi": 8.0}
    def simulate(theta, r):
        return count_summary(count_surrogate_sample(theta, calib, 300, r))
    res = run_sbc(simulate, prior=(0.2, 3.0), n_sbc=80,
                  abc_kwargs=dict(n_particles=200, n_generations=4),
                  rng=np.random.default_rng(0))
    assert res["chi2_p"] > 0.01     # uniform (loose bound for n_sbc=80)
```

- [ ] **Step 2: Run -> FAIL** (then PASS once Task 3 `run_sbc` exists; if Task 3 done it may already pass — that's fine, this is a guard).

- [ ] **Step 3: Implement `scripts/run_sbc.py`** — load calibration, build the
calibrated `simulate`, call `run_sbc` with `n_sbc=150`, write
`reports/figures/pdmp-03/sbc_rank_histogram.html` (rank histogram + the chi2 p
and PASS/FAIL verdict).

- [ ] **Step 4: Run the CLI:** `.venv/bin/python scripts/run_sbc.py` -> figure + chi2_p printed.
- [ ] **Step 5: Commit**
```bash
git add scripts/run_sbc.py tests/test_sbc.py reports/figures/pdmp-03/sbc_rank_histogram.html
git commit -m "feat(pdmp-03): run_sbc rank-histogram uniformity gate on the calibrated surrogate"
```

---

## Task 7: run_ppc.py — posterior-predictive coverage + figure

**Files:**
- Create: `v2ecoli/inference/ppc.py`
- Create: `scripts/run_ppc.py`
- Test: `tests/test_ppc.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_ppc.py
import numpy as np
from v2ecoli.inference.ppc import ppc_coverage

def test_ppc_coverage_in_range():
    rng = np.random.default_rng(0)
    calib = {"mu0": 40.0, "phi": 8.0}
    # posterior tightly around 1.0 -> predictive should cover ~90% of held-out
    posterior = rng.normal(1.0, 0.03, size=300)
    cov = ppc_coverage(posterior, calib, n_obs=300, n_heldout=200,
                       theta_true=1.0, rng=rng)
    assert 0.75 <= cov <= 1.0
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement**
```python
# v2ecoli/inference/ppc.py
from __future__ import annotations
import numpy as np
from v2ecoli.inference.count_surrogate import count_surrogate_sample, count_summary

def ppc_coverage(posterior, calib, *, n_obs, n_heldout, theta_true, rng):
    """Fraction of held-out observed summaries whose [mean] component falls in
    the posterior-predictive 90% interval."""
    pp = np.array([count_summary(count_surrogate_sample(
        rng.choice(posterior), calib, n_obs, rng))[0] for _ in range(400)])
    lo, hi = np.quantile(pp, [0.05, 0.95])
    held = np.array([count_summary(count_surrogate_sample(
        theta_true, calib, n_obs, rng))[0] for _ in range(n_heldout)])
    return float(np.mean((held >= lo) & (held <= hi)))
```
```python
# scripts/run_ppc.py — load calibration + the recovery posterior, call
# ppc_coverage, write reports/figures/pdmp-03/ppc_coverage.html.
```

- [ ] **Step 4: Run -> PASS.** Then run the CLI.
- [ ] **Step 5: Commit**
```bash
git add v2ecoli/inference/ppc.py scripts/run_ppc.py tests/test_ppc.py reports/figures/pdmp-03/ppc_coverage.html
git commit -m "feat(pdmp-03): run_ppc posterior-predictive coverage check"
```

---

## Task 8: record pdmp-03 gates + findings (honest, evidenced)

**Files:**
- Modify: `workspace/investigations/v2ecoli-pdmp/studies/pdmp-03-inference/study.yaml`

- [ ] **Step 1:** For the tests whose scripts now EXIST and PASS their gate
(`abc-posterior-recovery-on-synthetic`, `sbc-calibration-uniform-rank`,
`ppc-coverage-95-pct`): change their status from `planned`/`stub` to the
ran/passed value the file's enum uses, and replace the "scripts/run_*.py does
not exist" notes with the real result (the CI/CHI2_P/coverage numbers from the
actual runs). DO NOT mark them met if the run did not pass — record the real
outcome.
- [ ] **Step 2:** Add `findings[]` entries (status `confirms`, with
`evidence.from_run` + `provenance.run_ids: [phase3-surrogate-calibration, abc-smc-recovery, sbc, ppc]`)
stating the real posterior-recovery CI, the SBC chi2 p-value (PASS/FAIL), and the
PPC coverage. Keep the honest caveat: validated on a WCM-CALIBRATED COUNT
SURROGATE, not the full WCM in the loop (deferred to Phase 4).
- [ ] **Step 3:** `.venv/bin/python -c "import yaml; yaml.safe_load(open('<study.yaml>'))"` to confirm it parses.
- [ ] **Step 4: Commit**
```bash
git add workspace/investigations/v2ecoli-pdmp/studies/pdmp-03-inference/study.yaml
git commit -m "feat(pdmp-03): record real ABC-SMC/SBC/PPC gate results + findings"
```

---

## Self-review notes
- **Spec coverage:** surrogate calibration (T1,T4), forward model (T1), ABC-SMC (T2), SBC (T3,T6), PPC (T7), recovery gate (T5), SBC gate (T6), PPC gate (T7), study/gates/findings (T8). All spec components covered.
- **Type consistency:** `calib` dict is `{mu0, phi, theta_ref}` throughout (T1/T4/T5/T7); `count_surrogate_sample(theta, calib, n_obs, rng)` and `count_summary(counts)->[mean,std]` used identically in T2/T5/T6/T7; `abc_smc(observed, simulate, prior, *, n_particles, n_generations, rng)` signature consistent T2/T3/T5/T6; `run_sbc(simulate, prior, *, n_sbc, abc_kwargs, rng)` consistent T3/T6.
- **Placeholder scan:** the test in T5 references a `scripts_phase3` shim then explicitly directs to drop it for `from v2ecoli.inference.recover import recover` — the implementer must use the latter (noted inline). No other placeholders.
- **Honesty guard (T8):** gates are marked met ONLY if the real run passed; the surrogate-not-full-WCM caveat is mandatory.
