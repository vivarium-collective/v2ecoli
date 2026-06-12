# dnaa-4 dnaA self-autoregulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Rashmi's dnaA self-autoregulation mechanism (dnaA-promoter occupancy represses dnaA transcription) onto the active Phase-2 box-binding model, run it vs a no-autoregulation control, verify it closes the V-tension (re-initiations → 0, DnaA peak < 800, DnaA mean in [300,800], ATP-fraction in [0.2,0.5]), and record it as the spine-backed **dnaa-4** study.

**Architecture:** A per-tick negative-feedback loop. `dnaa_box_binding.py` computes the bound fraction `f` of the 2-per-chromosome dnaA-promoter sites (`pool_label == POOL_PROMOTER_HIGH`) and publishes it on the existing `dnaa_hydrolysis` process-state port. `transcript_initiation.py` reads `f` and scales the dnaA TU (`TU00259[c]`) promoter init-probs by `(1 − s·f)` *after* Mechanism-A rescaling. Unbound promoter (low DnaA) → full transcription; saturated promoter (high DnaA) → ~5× repression at `s = 0.8`. Backward-compatible: composites without the port wired see `f = 0` → factor 1 → no-op. The high-affinity `K_d` is loosened to 3 nM so the promoter genuinely de-occupies at low DnaA (the titration signal).

**Tech Stack:** v2ecoli (process-bigraph whole-cell model), `scripts/run_condition_multigen_parquet.py` (multigen runner), the run/outcome spine (`pbg_superpowers.study_evaluator` / `study_outcomes`, `pbg_emitters.RunReader`), the Mac mini for the 16-gen runs, the vivarium-dashboard for the study surface.

**Working dir:** `/Users/eranagmon/code/v2e-invest`, branch `investigation/dnaa-replication-v3` (now 0 behind framework main).

---

## Provenance & coordination

The autoregulation code is **Rashmi's** (handoff `~/Downloads/dnaa4_autoregulation_handoff.md`), **uncommitted** — her branch `feat/aim2-dnaa-oric-box-binding` tops out at the box-binding commit `4fe5cde` with no autoreg commit. **Her branch is protected — never rewrite/force-push it.** We re-apply her snippets as new commits on the investigation branch, crediting her in the commit body (`Co-authored-by` Rashmi if her address is known; otherwise note "mechanism by Rashmi, handoff <date>"). The box-binding substrate this builds on (`4fe5cde`) is already merged into v3 (PR #162), so only the +12/+25 autoreg lines are new here.

## Known unknowns (resolve as encountered; fallbacks given)

1. **Cache format.** The handoff's cache-patch recipe edits `out/<cache>/sim_data_cache.dill` → `obj["configs"][...]["perturbations"]` — the **older vEcoli dill** layout. v2ecoli's box-binding cache (`out/cache_dnaa3_sweep` on the mini) is the `save_sim_input` bundle (`initial_state.json` + `metadata.json` + the box catalog). **Task 3 resolves which path applies** and uses `--perturbation` on the runner (proven for dnaa-1/3) instead of dill-patching when the bundle format is in play.
2. **Steady-state start dill.** `out/steady_state_inputs/succinate_default_gen3_start_dnaa3.dill` is **missing**; a generic `succinate_default_gen3_start.dill` exists but predates box-binding. **Task 4** either generates a box-binding burn-in or runs from gen 1 (no `--resume-dill`) — the latter is simpler and the handoff's metrics are read from steady-state gens (≥ gen 3) regardless.
3. **K_h in the cache.** The handoff hard-patches `rates_fwd[29] = 7.51e-6` for `k_h = 0.025/min`. Confirm the equilibrium-rate index against the current cache before patching (Task 3) — an off-by-one silently mis-sets ATP-fraction.

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `v2ecoli/steps/dnaa_box_binding.py` | compute + publish `promoter_fraction` `f`; loosen `KD_HIGH_M` to 3 nM | Modify (+~13 lines) |
| `v2ecoli/processes/transcript_initiation.py` | new `dnaa_hydrolysis` port; constants; post-Mechanism-A scaling of dnaA TU | Modify (+~26 lines) |
| `tests/test_dnaa_autoregulation.py` | unit tests for `f` computation + the `(1−s·f)` scaling | Create |
| `out/cache_dnaa4_autoreg/` | box-binding cache (V=0.70e-3, k_h=0.025, K_d=3nM) | Assemble (Task 3) |
| `studies/dnaa-4-autoregulation/study.yaml` | spine study: mechanism, runs (autoreg+control), behavior tests, outcomes | Modify (exists as scaffold) |
| `studies/dnaa-4-autoregulation/charts/` | DnaA pool, ATP-fraction, promoter-occupancy-swing, autoreg-vs-control | Create (Task 8) |
| `scripts/render_dnaa4_autoreg.py` | chart + verification-metric renderer | Create (Task 7/8) |
| `investigations/dnaa-replication/feedback/<ts>.yaml` | response to Rashmi's handoff | Create (Task 9) |

---

## Task 1: Apply the `dnaa_box_binding.py` autoregulation + K_d loosening

**Files:**
- Modify: `v2ecoli/steps/dnaa_box_binding.py`
- Test: `tests/test_dnaa_autoregulation.py`

- [ ] **Step 1: Write the failing test for `f` computation**

```python
# tests/test_dnaa_autoregulation.py
import numpy as np

def test_promoter_fraction_from_pool_state():
    from v2ecoli.steps.dnaa_box_binding import _promoter_fraction, POOL_PROMOTER_HIGH, FORM_FREE
    # 2 promoter sites among 5 boxes; 1 of the 2 bound
    pool_label   = np.array([POOL_PROMOTER_HIGH, POOL_PROMOTER_HIGH, 0, 0, 0], dtype=np.int8)
    bound_form   = np.array([FORM_FREE, 1, FORM_FREE, 1, 1], dtype=np.int8)
    assert _promoter_fraction(pool_label, bound_form) == 0.5
    # zero promoter sites -> 0.0 (no divide-by-zero)
    assert _promoter_fraction(np.zeros(5, np.int8), bound_form) == 0.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dnaa_autoregulation.py::test_promoter_fraction_from_pool_state -v`
Expected: FAIL — `ImportError: cannot import name '_promoter_fraction'`.

- [ ] **Step 3: Add the helper + K_d loosening to `dnaa_box_binding.py`**

At the top, set the loosened high-affinity K_d (find the existing `KD_HIGH_M` / `K_d_high` constant; change `1e-9` → `3e-9`):

```python
KD_HIGH_M = 3e-9    # 3 nM — loosened so the promoter de-occupies at low DnaA
                    # (the titration signal that drives autoregulation)
```

Add the helper (pure, unit-testable; `POOL_PROMOTER_HIGH` / `FORM_FREE` already exist from the box-binding schema):

```python
def _promoter_fraction(pool_label, bound_form) -> float:
    """Bound fraction of the dnaA-promoter sites — the autoregulation signal."""
    prom_mask = (pool_label == POOL_PROMOTER_HIGH)
    n_total = int(prom_mask.sum())
    if n_total == 0:
        return 0.0
    n_bound = int((bound_form[prom_mask] != FORM_FREE).sum())
    return float(n_bound) / float(n_total)
```

- [ ] **Step 4: Publish `f` on the `dnaa_hydrolysis` port (after the Langmuir solve, before the `update` dict is returned)**

```python
        # ── dnaA self-autoregulation signal (mechanism by Rashmi, handoff 2026-06) ──
        promoter_fraction = _promoter_fraction(pool_label, new_bound_form)
        update["dnaa_hydrolysis"]["promoter_fraction"] = promoter_fraction
```

(Match the local variable names already in scope — `pool_label`/`new_bound_form` per the handoff; rename to the file's actual symbols if they differ.)

- [ ] **Step 5: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dnaa_autoregulation.py::test_promoter_fraction_from_pool_state -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/steps/dnaa_box_binding.py tests/test_dnaa_autoregulation.py
git commit -m "feat(dnaa-4): publish dnaA-promoter occupancy + loosen K_d to 3 nM

Mechanism by Rashmi (handoff 2026-06). dnaa_box_binding computes the bound
fraction of the POOL_PROMOTER_HIGH sites and publishes it on the dnaa_hydrolysis
port for transcript_initiation to read; K_d_high 1->3 nM so the promoter de-occupies."
```

---

## Task 2: Apply the `transcript_initiation.py` autoregulation scaling

**Files:**
- Modify: `v2ecoli/processes/transcript_initiation.py`
- Test: `tests/test_dnaa_autoregulation.py`

- [ ] **Step 1: Write the failing test for the scaling factor**

```python
# append to tests/test_dnaa_autoregulation.py
def test_autoreg_scaling_factor():
    from v2ecoli.processes.transcript_initiation import _autoreg_factor
    # s=0.8: f=0 -> 1.0 (full), f=1 -> 0.2 (repressed ~5x), f=0.5 -> 0.6
    assert _autoreg_factor(0.0, 0.8) == 1.0
    assert abs(_autoreg_factor(1.0, 0.8) - 0.2) < 1e-9
    assert abs(_autoreg_factor(0.5, 0.8) - 0.6) < 1e-9
    # strength 0 disables
    assert _autoreg_factor(1.0, 0.0) == 1.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dnaa_autoregulation.py::test_autoreg_scaling_factor -v`
Expected: FAIL — `ImportError: cannot import name '_autoreg_factor'`.

- [ ] **Step 3: Add the constants + helper + wire the port**

Constants near the top:

```python
DNAA_TU_IDX = 2778        # TU00259[c] — do not change unless the cache is rebuilt
AUTOREG_STRENGTH = 0.8    # s; 0 disables, 1 fully silences at f=1

def _autoreg_factor(promoter_fraction: float, strength: float) -> float:
    """Linear repression (1 - s*f). f in [0,1]."""
    return 1.0 - strength * promoter_fraction
```

Add the port to `TOPOLOGY`:

```python
TOPOLOGY = {
    ...
    "dnaa_hydrolysis": ("process_state", "dnaa_hydrolysis"),
}
```

In `update()`, **after** the Mechanism-A `_rescale_initiation_probs` call:

```python
        promoter_fraction = float(
            states.get("dnaa_hydrolysis", {}).get("promoter_fraction", 0.0))
        if promoter_fraction > 0.0 and AUTOREG_STRENGTH > 0.0:
            dnaa_promoters = (TU_index == DNAA_TU_IDX)
            if dnaa_promoters.any():
                self.promoter_init_probs[dnaa_promoters] *= _autoreg_factor(
                    promoter_fraction, AUTOREG_STRENGTH)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dnaa_autoregulation.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Confirm `DNAA_TU_IDX` against the cache (guard against silent mis-indexing)**

Run (once the cache is assembled in Task 3):
```bash
.venv/bin/python -c "from v2ecoli... load sim_data; print(sim_data.process.transcription.rna_data['id'][2778])"
```
Expected: contains `TU00259`. If not, correct `DNAA_TU_IDX` and re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/processes/transcript_initiation.py tests/test_dnaa_autoregulation.py
git commit -m "feat(dnaa-4): scale dnaA transcription by (1 - s*f) from promoter occupancy

Mechanism by Rashmi (handoff 2026-06). After Mechanism-A rescaling, the dnaA TU's
promoter init-probs are scaled by (1 - 0.8*f). Backward compatible: no port -> f=0 -> no-op."
```

---

## Task 3: Assemble the box-binding cache (V=0.70e-3, k_h=0.025, K_d=3 nM)

**Files:**
- Create: `out/cache_dnaa4_autoreg/`

The autoregulation requires the **box catalog** (DnaA_box sites incl. `POOL_PROMOTER_HIGH`). The dnaa-2 caches are pre-box-binding — unusable. The box-binding cache lives on the mini at `out/cache_dnaa3_sweep`.

- [ ] **Step 1: Pull the box-binding cache from the mini**

```bash
rsync -a mini:code/v2e-invest/out/cache_dnaa3_sweep/ out/cache_dnaa4_autoreg/
```

- [ ] **Step 2: Verify the box catalog + promoter pool are present**

```bash
.venv/bin/python -c "
from v2ecoli.core import load_sim_input
sd = load_sim_input('out/cache_dnaa4_autoreg')
mc = sd.process.replication.motif_coordinates or {}
print('promoter pool present:', any('promoter' in k.lower() for k in mc))
print('box pools:', list(mc.keys())[:8])
"
```
Expected: `promoter pool present: True`.

**Fallback** (if the mini cache is gone): rebuild from the box-catalog ParCa in the adopt worktree — `rsync -a mini:.../sim_data_dnaa3/ out/sim_data_dnaa4/` then `save_sim_input(... condition='succinate', fixed_media='minimal_succinate')` (the Task-3 build recipe from the dnaa-3 sweep, memory `reference_v2ecoli_internals_map`).

- [ ] **Step 3: Decide the V / k_h injection path (resolves Unknown #1 + #3)**

The runner takes `--perturbation 'TU00259[c]=<V>'` (proven on dnaa-1/3) — prefer this over dill-patching. For `k_h`, locate the equilibrium hydrolysis rate in the cache:
```bash
.venv/bin/python -c "
from v2ecoli.core import load_sim_input
sd = load_sim_input('out/cache_dnaa4_autoreg')
# find the bound-pool hydrolysis reaction index; print the candidate rate
# (compare against handoff rates_fwd[29]=7.51e-6 for 0.025/min)
print([r for r in getattr(sd.process,'equilibrium',[]) ][:0] or 'inspect equilibrium rates_fwd here')
"
```
Record the resolved index; if the cache embeds k_h structurally (not via rates_fwd), document that the 0.025/min setpoint is already baked and no patch is needed.

- [ ] **Step 4: Commit the decision (cache is gitignored; commit the recipe note)**

```bash
echo "cache_dnaa4_autoreg: cache_dnaa3_sweep + V via --perturbation 0.70e-3, k_h=0.025, K_d=3nM (runtime). Built <date>." >> docs/superpowers/plans/2026-06-12-dnaa4-autoregulation.md
git add docs/superpowers/plans/2026-06-12-dnaa4-autoregulation.md
git commit -m "docs(dnaa-4): record cache_dnaa4_autoreg assembly recipe"
```

---

## Task 4: Resolve the steady-state start (resolves Unknown #2)

**Files:** none created — a decision + verification.

- [ ] **Step 1: Decide start strategy**

The dnaa3-specific burn-in dill is missing. Two options:
- **A (preferred, simpler):** run from gen 1 (omit `--resume-dill`/`--start-gen`); read metrics from steady-state gens (≥ gen 3), as the handoff's own acceptance windows do.
- **B:** generate a box-binding burn-in: a 3-gen run to `out/steady_state_inputs/succinate_box_gen3_start.dill`, then resume. Only if A's early gens contaminate the 16-gen statistics.

Default to **A**. Record the choice in the plan doc.

- [ ] **Step 2: No commit** (decision recorded in Task 3's doc append).

---

## Task 5: Run the no-autoregulation control (on the mini)

**Files:** produces `out/dnaa4_control_s0_16gen/` (mini, rsynced back).

- [ ] **Step 1: Push the autoreg code + cache to the mini**

```bash
git push origin HEAD:investigation/dnaa-replication-v3
ssh mini-cmd 'cd ~/code/v2e-invest && git fetch origin -q && git reset --hard origin/investigation/dnaa-replication-v3'
rsync -a out/cache_dnaa4_autoreg/ mini:code/v2e-invest/out/cache_dnaa4_autoreg/
```

- [ ] **Step 2: Set `AUTOREG_STRENGTH = 0.0` for the control and launch 16 gens**

On the mini, temporarily export the disable (don't edit the committed default — pass via env if supported, else a throwaway local edit reverted after):
```bash
ssh mini-cmd 'cd ~/code/v2e-invest && PYTHONPATH=. nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa4_autoreg --out-dir out/dnaa4_control_s0_16gen \
  --experiment-id dnaa4_control_s0_16gen --generations 16 --max-min 180 --seed 0 \
  --perturbation "TU00259[c]=0.70e-3" > /tmp/dnaa4_control.log 2>&1 &'
```
(If `AUTOREG_STRENGTH` can't be env-overridden, add a `--autoreg-strength` CLI flag to the runner as a tiny prerequisite step — preferred over a throwaway edit.)

- [ ] **Step 3: Wait + verify completion** via parquet count, not the buffered log (memory `reference_mini_headless_agents`).

- [ ] **Step 4: rsync the control back**

```bash
rsync -a mini:code/v2e-invest/out/dnaa4_control_s0_16gen/ out/dnaa4_control_s0_16gen/
```

---

## Task 6: Run the autoregulation experiment (on the mini)

**Files:** produces `out/dnaa4_autoreg_s08_16gen/`.

- [ ] **Step 1: Launch 16 gens with `AUTOREG_STRENGTH = 0.8` (the committed default)**

```bash
ssh mini-cmd 'cd ~/code/v2e-invest && PYTHONPATH=. nohup .venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa4_autoreg --out-dir out/dnaa4_autoreg_s08_16gen \
  --experiment-id dnaa4_autoreg_s08_16gen --generations 16 --max-min 180 --seed 0 \
  --perturbation "TU00259[c]=0.70e-3" > /tmp/dnaa4_autoreg.log 2>&1 &'
```

- [ ] **Step 2: Wait + verify** (parquet count) **+ rsync back** to `out/dnaa4_autoreg_s08_16gen/`.

---

## Task 7: Verify the expected results

**Files:**
- Create: `scripts/render_dnaa4_autoreg.py`

- [ ] **Step 1: Write the metric extractor (RunReader-based, both runs)**

```python
# scripts/render_dnaa4_autoreg.py  (metrics first; charts in Task 8)
from pbg_emitters.run_reader import RunReader

DNAA_BULK = ["PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"]  # apo+ATP+ADP

def metrics(store):
    rr = RunReader.open(store)
    gens = rr.generations()
    # total DnaA per gen (bulk+bound), ATP fraction, re-init count, peak
    # (reuse the dnaa-3 readers in scripts/render_dnaa3_feedback_charts.py)
    ...
    return {"gens": len(gens), "dnaa_peak": ..., "dnaa_mean_band": ...,
            "atpfr_range": ..., "reinit_count": ...}
```
(Port the per-pool/total readers already written in `scripts/render_dnaa3_*` — DRY.)

- [ ] **Step 2: Run it on both runs + assert the acceptance criteria**

```bash
.venv/bin/python scripts/render_dnaa4_autoreg.py --autoreg out/dnaa4_autoreg_s08_16gen --control out/dnaa4_control_s0_16gen
```
Expected (vs the handoff targets):
- **re-initiations → 0** across 16 gens (control had ≥1 at gen 5).
- **DnaA peak < 800** across all gens (control peaked ~847).
- **DnaA mean in [300,800]** every steady-state gen.
- **ATP-fraction in [0.2,0.5]** (k_h=0.025 already handles this).
- **Promoter occupancy 50%→100%→50% within-cycle swing**, transcription `V·(1−0.8·f)` riding inversely.

- [ ] **Step 3: If any criterion fails → branch to Task 9 (linear vs Hill)** before declaring the mechanism validated.

- [ ] **Step 4: Commit the metric script**

```bash
git add scripts/render_dnaa4_autoreg.py
git commit -m "feat(dnaa-4): autoreg-vs-control metric extractor + acceptance checks"
```

---

## Task 8: Record the dnaa-4 study via the run/outcome spine

**Files:**
- Modify: `studies/dnaa-4-autoregulation/study.yaml`
- Create: `studies/dnaa-4-autoregulation/charts/*.png` (+ `.meta.json`)

- [ ] **Step 1: Render the charts** (`render_dnaa4_autoreg.py --charts`): `dnaa4_pool_band`, `dnaa4_atp_fraction`, `dnaa4_promoter_swing` (the within-cycle f + transcription-rate trace), `dnaa4_autoreg_vs_control` (peak/re-init overlay). Each with a `.meta.json` (`source_run_id`, caption).

- [ ] **Step 2: Wire the runs into `study.yaml`** — two `runs[]` entries (`dnaa4_autoreg_s08_16gen` with `emitter.store`, and `dnaa4_control_s0_16gen`), so the framework-baked run summary (generations / sim_minutes / n_readouts) records via `study_outcomes.record_runs` and the Simulations tab renders them.

```bash
.venv/bin/python -c "from pbg_superpowers.study_outcomes import record_runs; print(record_runs('studies/dnaa-4-autoregulation'))"
```

- [ ] **Step 3: Author the behavior tests** (`tests:` list with `measure`/`pass_if`, code-evaluable):
- `reinit-events-zero` — re-init count `== 0` over the lineage.
- `dnaa-peak-under-800` — peak total DnaA `<= 800` every generation.
- `dnaa-pool-within-band` — generation-average total DnaA `in_range [300,800]`.
- `dnaa-atp-fraction-in-band` — ATP fraction `in_range [0.2,0.5]`.
- `promoter-occupancy-swings` (agent-routed — trajectory-over-cycle).

- [ ] **Step 4: Compute outcomes via the evaluator**

```bash
.venv/bin/python -c "from pbg_superpowers.study_evaluator import compute_outcomes; print(compute_outcomes('studies/dnaa-4-autoregulation', '.'))"
```
Expected: the 4 code tests resolve PASS (if Task 7 passed) with `reconcile` flags vs authored; the promoter-swing routes to agent.

- [ ] **Step 5: Set the study status + objective + conclusion** (the autoregulation closes the V-tension the V-sweep proved), then commit.

```bash
git add studies/dnaa-4-autoregulation/study.yaml studies/dnaa-4-autoregulation/charts/
git commit -m "study(dnaa-4): autoregulation runs + behavior tests + computed outcomes"
```

- [ ] **Step 6: Refresh the dashboard + open** `:8771/studies/dnaa-4-autoregulation` — confirm the Simulations tab shows real sim-time/seeds/runs, the Tests tab shows the computed outcomes, and the charts render.

---

## Task 9: Decide linear vs Hill repression; respond to Rashmi

**Files:**
- (conditional) Modify: `v2ecoli/processes/transcript_initiation.py`
- Create: `investigations/dnaa-replication/feedback/<ts>.yaml`

- [ ] **Step 1: Evaluate the open question.** If Task 7 met all criteria at `s = 0.8` linear → keep linear; record the decision. If DnaA still over-shoots mid-cycle → implement the Hill form (the handoff's open question):

```python
def _autoreg_factor(f, strength, n=4, K=0.5):
    # Hill: sharper switch — less repression below f~0.5, more above
    return 1.0 - strength * (f**n) / (K**n + f**n)
```
Re-run Task 6/7 with the Hill form and compare peak/re-init.

- [ ] **Step 2: Write the inline response to Rashmi's handoff** — what was integrated, the autoreg-vs-control result (re-inits, peak, band, ATPfr), the linear-vs-Hill decision, and any tuning. Commit + push.

- [ ] **Step 3: Final dashboard refresh + summary to Eran.**

---

## Self-review notes

- **Spec coverage:** code application (T1–T2), missing inputs (T3 cache, T4 dill), control + experiment runs (T5–T6), verification of all five handoff criteria (T7), spine study with runs+tests+outcomes+charts (T8), linear-vs-Hill decision + Rashmi response (T9). All handoff sections mapped.
- **Decisions deferred to execution (flagged, with fallbacks):** cache format / V·k_h injection (T3), start-dill strategy (T4), `AUTOREG_STRENGTH` override path for the control (T5 — prefer a `--autoreg-strength` runner flag over a throwaway edit), linear vs Hill (T9).
- **Reproducibility caveat:** the framework merge (just done) may shift sim numerics vs Rashmi's original; T7's control run is the apples-to-apples baseline (same framework), so the autoreg-vs-control comparison is internally valid regardless.
- **Heavy steps:** the two 16-gen runs (T5/T6) are the long poles — mini, verify via git/parquet not logs.
