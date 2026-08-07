# ParCa apparatus — deep review (robustness / performance / caching)

**Scope:** `v2ecoli/cli/parca.py`, `v2ecoli/processes/parca/**`, `v2ecoli/library/cache_version.py`,
`v2ecoli/core.py` (bundle writer/reader), `scripts/build_cache.py`, `scripts/build_condition_cache.py`,
`.github/workflows/ci.yml`, `v2ecoli/comparison/vecoli_parca.py`.
**Worktree:** `/Users/eranagmon/code/v2ecoli--parca-review` @ `c53017b7` (main, incl. PR #445).
**Mode:** read-only review. **No code was modified.**

---

## Executive summary

The ParCa *fit* is structurally sound — a clean 9-step process-bigraph DAG, worker functions
that return dicts instead of mutating `sim_data`, deterministic seeding (`range(N_SEEDS)`),
`sorted()` on every load-bearing iteration, no `deepcopy` anywhere, and a working
resume-from-checkpoint path. The **apparatus around it has decayed**, and its two headline
guarantees are both currently inoperative: five of the eleven files in the cache
fingerprint's `INPUT_FILES` **no longer exist** (renamed in `645fe178`, never updated), so
they hash to a constant `"MISSING"` sentinel and the composite half of the fingerprint is
inert; and **no production code path calls `verify_cache_version` at all** — the sole caller
in the repo is a pytest fixture — despite `AGENTS.md:287` asserting that `build_composite`
does. On top of that, two independent `print-and-continue` paths (`step_09_final_adjustments.py:158-165`
and `core.py:197-216`) let a **partially-fit** ParCa and an **incomplete** cache bundle be
written and fingerprinted as if they were complete.

**Top 3 highest-leverage changes** (all safe; none alters calibration output):

1. **Repair the fingerprint, and make a missing input a hard error** — `cache_version.py:56-60`
   + `ci.yml:101-105`, plus two guard tests. Restores the module's entire reason for existing. ~1 h.
2. **Actually call `verify_cache_version` on the load path** — `v2ecoli/core.py:126-167`.
   Today the guard is dead code outside pytest. ~1 h.
3. **Fail loudly instead of printing** — `step_09:158-165` and `core.py:197-216`. A ParCa that
   couldn't fit the mechanistic kcats, or a bundle missing `ecoli-metabolism`, must not be
   indistinguishable from a good one. ~2 h.

Performance is genuinely secondary — full ParCa is **~2.5 min measured** (`models/parca/runtimes.json`
Σ = 153.5 s; `workspace/studies/showcase-1-parca/study.yaml:99` records 142.4 s), not the
"4–8 hours" the docs still claim in eight places. But there are three large, *safe* wins:
an uncached `stoich_matrix()` rebuilt inside a triple-nested loop, a 634 MiB `sim_data`
pickled once **per condition** to spawn workers, and 90 MB of every 266 MB cache bundle that
nothing on the v2ecoli path reads.

---

## The apparatus in one picture

There are **two** distinct caches, and the naming blurs them:

```
  ecoli-sources TSVs (~140 files, unpinned pkg)  +  parca_overrides.tsv (implicit)
                            │
                            ▼
  [ LAYER 1 — the fit ]  v2ecoli-parca  →  9 Steps (tick-serialized)  →  parca_state.pkl
      ├─ inputs: raw KB, --mode fast|full, --cpus, --no-operons, --bundle-manifest-path,
      │          V2PARCA_N_SEEDS, SLURM_CPUS_*, scipy/numpy/cvxpy/ecos/stochastic-arrow builds
      ├─ intra-step cache: parca-km-<crc32>.cPickle   ← step 3 ONLY
      └─ NO fingerprint. NO provenance. NO content address. Partial fits look complete.
                            │
                     (gzip + git commit)
                            ▼
       models/parca/parca_state.pkl.gz        ← the de-facto calibration
       38,038,445 B gzipped / 664,887,631 B raw
                            │
  [ LAYER 2 — the bundle ]  ▼
      scripts/build_cache.py → out/cache/{initial_state.json  10.4 MB,
                                          sim_data_cache.dill 164.9 MB,
                                          simData.cPickle      90.2 MB,   ← unread by v2ecoli
                                          metadata.json, cache_version.json}
      └─ fingerprint: cache_version.INPUT_FILES (11 paths, 5 of which do not exist)
      └─ verified by: nothing on the production path
```

`cache_version.py` guards **Layer 2 only**, and correctly so — Layer 2 is a pure function of
the fixture plus the `LoadSimData` boundary. The structural gap is that **Layer 1 has no
equivalent at all**: nothing records which code, which mode, which env, or which dependency
versions produced the fixture that Layer 2 so carefully attests to.

**Measured step profile** (from `models/parca/runtimes.json`, full mode, 51 conditions):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **Σ** |
|---|---|---|---|---|---|---|---|---|---|---|
| s | 5.0 | 0.1 | 11.0 | **57.7** | **52.0** | **20.8** | 0.0 | 0.1 | 6.8 | **153.5** |

Steps 4+5+6 = **85%** of wall time. Note the checkpoint-pickling time is *excluded* from these
numbers (the timer stops at `cli/parca.py:126`, before the dump at `:133`), so real wall time
is higher than 153.5 s — see A24.

---

## Prioritized findings

Ordered by impact ÷ risk. **Tier A is safe — nothing in it can change calibration output**
(A22 and A25 change *which* value is used only in already-degenerate cases; both are flagged).
Tier B can move numbers and needs the verification plan at the end.

### Tier A — safe (no calibration change)

#### A1 · **Critical** · caching-correctness · `v2ecoli/library/cache_version.py:56-60`

**Five of eleven `INPUT_FILES` name files that no longer exist.** Verified:

```
OK   models/parca/parca_state.pkl.gz          MISS v2ecoli/composites/baseline.py
OK   v2ecoli/library/sim_data.py              MISS v2ecoli/composites/baseline_population.py
OK   v2ecoli/library/unit_bridge.py           MISS v2ecoli/composites/baseline_time_varying_env.py
OK   v2ecoli/types/quantity.py                MISS v2ecoli/composites/colony.py
OK   v2ecoli/library/initial_conditions.py    MISS v2ecoli/composites/baseline_millard.py
OK   v2ecoli/core.py
```

The composite family was renamed to `ecoli_*` in `645fe178` ("rename: ecoli_ scheme for the
whole-cell composite family"); `INPUT_FILES` was never updated, despite `AGENTS.md:157`
explicitly requiring it as step 3 of "Adding a new composite architecture."

*Failure:* `compute_cache_version` maps each to the **stable** string `"MISSING"`
(`cache_version.py:124-128`), so edits to the real composites no longer move `inputs_hash`
**at all**. Change the document shape in `ecoli_baseline.py` → a cache built against the old
architecture is served silently. That is verbatim the hazard the comment at `:53-55` claims
to prevent. It also defeats the careful `_default_repo_root` reasoning at `:102-113` by a
different door: that comment worries about the fingerprint "collaps[ing] to a constant that
never changes when the source changes" — which is exactly what has happened, for 5 of 11 inputs.

*Fix:* repoint to `ecoli_baseline.py`, `ecoli_population.py`, `ecoli_time_varying_env.py`,
`ecoli_colony.py`, `ecoli_millard.py`; add `v2ecoli/composites/_helpers.py` (the shared
builder, currently untracked). **Make a missing input `raise`** rather than encode `"MISSING"` —
a vanished fingerprint input is a bug, not a state. Bump `SCHEMA_VERSION`. · **~30 min**

#### A2 · **Critical** · caching-correctness · `v2ecoli/core.py:126-167`

**No production loader verifies the cache.** `load_cache_bundle` and
`_load_cache_bundle_cached` never call `verify_cache_version`; neither does `build_composite`
(`v2ecoli/__init__.py:15-63`). The only non-self-test caller in the entire repo is
`tests/conftest.py:155`.

*Failure:* every one of `ecoli_baseline.py:587,928`, `ecoli_millard.py:530`,
`millard_fba_bridge_harness.py:246`, `scripts/run_condition_multigen_parquet.py:348,432`,
`pbg_v2ecoli/uq_sim_data_injection.py:220,248`, and the eleven test modules that hardcode
`CACHE = "out/cache"` loads **unverified**. Pull a branch that changed `sim_data.py`, run a
sim, get a silently mis-calibrated result — the 10-frame `AttributeError` scenario in
`cache_version.py:1-15`, undefended. `AGENTS.md:287` states the opposite and is factually wrong.

*Fix:* `verify_cache_version(cache_dir)` at the top of `load_cache_bundle`, with a
`V2ECOLI_SKIP_CACHE_VERIFY=1` escape hatch for deliberate cross-version work. Correct
`AGENTS.md:287`. · **~1 h**

#### A3 · **Critical** · robustness · `v2ecoli/processes/parca/steps/step_09_final_adjustments.py:149-165`

**A partially-fit ParCa is written as if complete.** The three mechanistic fits are wrapped
in a `for label, call in [...]` loop with `try: call() except Exception as e: print(...)`:

```python
            try:
                call()
            except Exception as e:
                print(f"  Step 9 WARNING: {label} failed ({type(e).__name__}: {e}); "
                      "continuing so the pipeline produces a comparable pickle.")
```

*Failure:* if `set_mechanistic_supply_constants` / `_export_` / `_uptake_` raises (the comment
at `:139-147` says this happens on "numerically-marginal kinetics … in debug mode where the
truncated TF set produces edge-case input distributions"), the pipeline **still writes
`parca_state.pkl`**. That pickle is byte-shaped exactly like a complete one, carries no marker,
and the only evidence is a `print` on stdout that the CLI captures into a `StringIO`
(`cli/parca.py:150,166`) and then discards apart from a runtime regex. Six months later
nobody can tell whether the shipped fixture has mechanistic uptake constants or not. Note this
is the *same* failure mode as the NNLS fallback in B1 — both are "the marginal case fired,
silently."

*Fix:* record `{label: ok|error}` for all three into the composite state and into the
provenance sidecar (A11); refuse to write `parca_state.pkl` on failure unless
`--allow-partial-fit` is passed. · **~2 h**

#### A4 · High · caching-correctness · `.github/workflows/ci.yml:92-106` vs `cache_version.py:37-61`

The CI `hashFiles` list and `INPUT_FILES` disagree on one name: CI hashes
`v2ecoli/composites/millard_pdmp_baseline.py` (`:105`), the code hashes
`v2ecoli/composites/baseline_millard.py` (`:60`). Neither exists, so today they *accidentally*
agree. `restore-keys` is absent (verified) — so prefix-fallback restore is **not** an
additional risk.

*Failure:* the instant either name is created, the lists diverge. Create `baseline_millard.py`
→ the in-code hash moves, the CI key does not → `actions/cache` reports a hit → the "Build ParCa
cache on miss" step (`ci.yml:108-110`) is **skipped** → every sim test fails at
`conftest.py:155` with `StaleCacheError`, with no way to bust the key short of bumping
`parca-cache-v4` by hand.

*Fix:* derive the CI list from `INPUT_FILES` (emit it into `$GITHUB_OUTPUT`), or add
`test_ci_key_matches_input_files` that parses `ci.yml` and asserts set equality. Add
`test_input_files_all_exist`. **Both tests fail on `main` today** — that is the point. · **~1 h**

#### A5 · High · robustness · `v2ecoli/core.py:185-256`

**The bundle is written non-atomically, in place.** Order: `initial_state.json` (:195) →
`sim_data_cache.dill` (:229) → `simData.cPickle` (:239) → `metadata.json` (:253) →
`cache_version.json` (:255). No temp dir, no `os.replace`, no fsync.

*Failure:* the ordering is right for a *fresh* build (marker last → an interrupt leaves no
marker → `StaleCacheError` "or was partially written"). But a **rebuild over an existing
`out/cache`** leaves the *previous, valid* `cache_version.json` next to a truncated 165 MB
`sim_data_cache.dill`, and `verify_cache_version` then **passes on corrupt data**. Trigger:
SIGINT, OOM, or the 30-min CI job cap (`ci.yml:61`) during the dill write.

*Fix:* two lines — `os.remove(cache_version.json)` as the *first* action of
`_write_sim_input_bundle`. Better: write into `<dir>.tmp/` and `os.replace` the directory. · **1–3 h**

#### A6 · High · robustness · `v2ecoli/core.py:197-216`

**An incomplete bundle is fingerprinted as valid.** Per-config build failures are collected
and **printed**, then the bundle is written anyway and `write_cache_version` (:255) stamps it.
The fingerprint attests to *inputs*, never to *completeness*.

*Failure:* a bundle missing `ecoli-mass-listener` or `ecoli-metabolism` passes verification;
the sim then dies on a divide-by-zero in `listeners.mass.cell_mass` — the failure the comment
at `:203-210` documents. Under `pytest -n auto` (`ci.yml:126`) the print is buried in
interleaved worker output and is effectively invisible. This is A3's twin, one layer down.

*Fix:* record `sorted(configs)` into `CacheVersion`; have `verify_cache_version` assert the
expected set. At minimum hard-fail on a required subset. · **~2 h**

#### A7 · High · caching-correctness · `v2ecoli/core.py:271-294`; `scripts/build_condition_cache.py:206-208`

**Different fits collide on one key.** `save_sim_input(..., seed, condition, fixed_media)` and
the *patched* sim_data in `build_condition_cache.py` all produce bundles whose
`cache_version.json` carries an **identical** `inputs_hash` — none of those parameters is in
the fingerprint.

*Failure:* `out/cache` (basal) and `out/cache-stage1-heuristic` (dnaA-patched) are
indistinguishable to `verify_cache_version`. Point `--cache-dir` at the wrong one, or copy /
symlink between them, and the guard is silent. `core.py:243-247` *acknowledges* the media
hazard and writes `media_id` into `metadata.json` — but that field is checked in exactly one
ad-hoc place (`scripts/run_comparison_ensemble.py:133`), never on the general load path.

*Fix:* add a `build_params` block (`condition`, `fixed_media`, `seed`, patch id /
condition-manifest hash) to `CacheVersion`, fold into `inputs_hash`, bump `SCHEMA_VERSION`.
`build_condition_cache.py`'s existing `condition.json` manifest then becomes enforceable
rather than advisory. · **half day**

#### A8 · High · caching-correctness · `v2ecoli/processes/parca/steps/step_05_fit_condition.py:70`

**`V2PARCA_N_SEEDS` is an untracked fit input — confirmed.**

```python
import os as _os
N_SEEDS = int(_os.environ.get('V2PARCA_N_SEEDS', '10'))
```

Read **once at module import**, at module scope — not a Step config, not in `config_schema`,
not wired to any port. It is a genuine fit input, not just a speed knob: `:136-139` allocates
`(N_SEEDS, M)`, `:152`/`:164` seed `StochasticSystem`/`RandomState` from `range(N_SEEDS)`, and
`:230-233` reduces to `mean(0)`/`std(0)` → `bulkAverageContainer`/`bulkDeviationContainer`.
Those means propagate into `calculateTranslationSupply` (`:304`), into Step 6's `_build_matrix_H`
(`promoter_fitting.py:301-303`, which **zeroes an `H` entry when a TF's average count is 0** — a
*qualitative*, structural change to the convex program, not just noise), into Step 7's
`fitLigandConcentrations` (`promoter_fitting.py:610,620`), and into Step 9.

*Failure:* fits made with `V2PARCA_N_SEEDS=3` and `=10` differ in every condition's bulk
distribution, produce two different `parca_state.pkl`, and after `build_cache.py` produce two
`out/cache` bundles with **byte-identical `cache_version.json`**. Nothing in the artifact,
`runtimes.json`, or git records which one you have. The repo already knows this —
`v2ecoli/library/comparison_composite.py:202-215` folds it into the *comparison harness's*
cache address — but the mitigation never made it into `v2ecoli-parca` itself, so the harness's
notion of `n_seeds` can silently disagree with the fit's.

*Fix:* promote to `FitConditionStep.config_schema` with an env fallback; echo the resolved
value into the composite state and the provenance sidecar (A11). Keep the env var — just make
it visible. · **2–4 h**

#### A9 · High · caching-correctness · `pyproject.toml:33-48`; `_scipy_compat.py`; `cache_version.py:37-61`

**The fingerprint has zero coverage of the runtime environment**, and this repo has already
been burned by it twice and carries the scar tissue:

- The whole of `_scipy_compat.py` (137 lines) exists to rehydrate `PPoly`/`CubicSpline`
  pickled by a pre-1.15 scipy under a post-1.15 scipy.
- `data_loader.py:43-50` carries a *second* remap for `scipy._lib.array_api_compat.*` →
  `array_api_compat.*` (scipy ≥1.16 dropped the vendored copy).
- `pyproject.toml:39` pins `scipy<1.18` because 1.18 refactored `scipy.interpolate` — **with
  no floor**, so 1.11 and 1.17 are both permitted and behave differently (see B2/B3).
- `cvxpy` (`pyproject.toml:42`), `ecos`, and `stochastic-arrow` are entirely unpinned, and all
  three sit on the fit path.

All of these are *runtime* patches for what is really a *cache-key* problem. A cache built
under scipy 1.14 and loaded under scipy 1.17 passes `verify_cache_version` unchanged. Worse,
`_scipy_compat.install()` bails silently on any exception (`:41-42`, `:51-52`), so the next
scipy internals change turns the shim into a no-op and the deep `AttributeError` returns with
no signal that the bridge disengaged. And `install()` is called **only** from
`data_loader.py:23` — the *fit* path (`cli/parca.py`, `composite.py`) never installs it.

*Fix:* add `{"python", "scipy", "numpy", "numba", "dill", "cvxpy", "ecos", "stochastic-arrow"}`
versions to `CacheVersion` and to `inputs_hash`. **The correct pattern already exists in this
repo** — `v2ecoli/comparison/vecoli_parca.py:167-176` records exactly this `context` block, and
its docstring (`:12-19`) diagnoses this precise failure. Copy it inward. Add a scipy floor. · **2–4 h**

#### A10 · High · perf · `step_05_fit_condition.py:183`; `equilibrium.py:224-234` vs `complexation.py:149-161`

**`equilibrium.stoich_matrix()` is rebuilt from scratch inside a triple-nested loop.** At
`step_05:183`, inside the `while np.linalg.norm(metDiffs, inf) > 1` convergence loop (`:163`),
inside `for seed in range(N_SEEDS)` (`:139`), inside the per-condition worker:

```python
            bulkContainer["count"][equilibrium_molecules_idx] += np.dot(
                sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
                rxnFluxes.astype(np.int64),
            )
```

`equilibrium.stoich_matrix()` (`equilibrium.py:224-234`) allocates a dense `np.zeros(shape)` and
scatters on **every call**, with no caching — and then `.astype(np.int64)` allocates a second
full copy. Its sibling `complexation.stoich_matrix()` (`complexation.py:149-161`) **is**
memoized on the instance, with an explicit comment: *"The result is cached on the instance
because … repeated reconstruction [is] very expensive for large bulk_molecules lists (~16 k
entries in E. coli)."* Same reasoning, same class of matrix, opposite treatment.

*Cost:* 51 conditions × 10 seeds × up to ~100 iterations = up to ~51,000 dense allocations +
casts per run, inside the step that costs 52 s.

*Fix:* memoize identically to `complexation.py:158-161`, and hoist the `.astype(np.int64)` out
of the loop. Pure refactor — the matrix is built from immutable `_stoichMatrixI/J/V`. · **~1 h**

#### A11 · High · robustness / provenance · `models/parca/README.md`; commits `71442761`, `58353939`; `cli/parca.py:188-198`

**The shipped calibration is unreproducible and its own documentation contradicts it.**

- `README.md:1-7,49-53` says the fixture is `--mode fast`, 71.6 min, step 5 = 70 min, 7 conditions.
- `71442761` (2026-06-03) replaced it with a **full-mode** fit — "all 23 TFs / 51 conditions".
- `models/parca/runtimes.json`, in the same directory, totals **153.5 s**.
- Counting `*__active`/`*__inactive` tokens in the committed pickle yields **46 TF conditions**
  (+ basal/with_aa/acetate/succinate/no_oxygen = 51) — the full-mode count, not 7.
- `58353939` (2026-06-10) then **hand-edited the pickle** to add two `int8` fields to the
  `DnaA_box` dtype, shrinking it 41,246,029 → 38,038,445 B (−7.8%) while claiming only to *add*
  two fields. Plausibly a re-gzip at a different level — but **unverifiable**, which is the point.
- `README.md` has not been touched since `a484baf0` (2026-04-16).

Compounding it, the CLI records **no provenance whatsoever**: `cli/parca.py:188-198` strips
tick and step-slot keys from `composite.state` and pickles the rest. `--mode`, `--cpus`,
`--no-operons`, `--bundle-manifest-path`, `V2PARCA_N_SEEDS`, the producing git SHA, and the
dependency versions are all absent. Two pickles from `--mode fast` and `--mode full` are
**indistinguishable on disk**.

*Failure:* nobody — human or agent — can determine what the current fixture is. An agent
reading `README.md:1-7` will correctly conclude the shipped fixture is fast-mode and therefore
unfit for production, and be wrong. Re-running the pipeline today will not reproduce the
shipped bytes.

*Fix:* emit `models/parca/provenance.json` at generation: git SHA, `--mode`, `--cpus`,
`--no-operons`, bundle-manifest hash, `V2PARCA_N_SEEDS`, `PYTHONHASHSEED`, dependency versions,
SHA256 of the output, step-9 partial-fit status (A3), and `manual_patches: []`. Rewrite
`README.md:49-53`. Then regenerate cleanly so `manual_patches` is genuinely empty. · **3 h + one clean regen**

#### A12 · High · perf · `steps/_facade.py:79-82`; `step_05:395-398`; `step_04:135-142`; `parallelization.py:201-208`

**The entire 634 MiB `sim_data` is pickled once per condition to feed spawn workers.**
`apply_updates` (`fitting.py:972-978`) does `pool.apply_async(func, a)` — one task per
condition — and each task's args begin with the facade `sd`:

```python
        args = [(sd, working_specs[condition], condition) for condition in condition_labels]
```

`_FacadeProxy.__reduce__` (`_facade.py:79-82`) returns `(_FacadeProxy, (self._ns, self._root))`,
where `_root` is `sim_data_root` — the whole `SimulationDataEcoli`. The pool uses the **spawn**
start method (`parallelization.py:201-208`), so there is no fork COW sharing: at 51 conditions
in full mode that is **~51 full sim_data serializations per parallel step**, for steps 4 and 5
both (110 s of the 153 s total). Peak RSS ≈ `cpus` × sizeof(sim_data) plus the parent's copy.

*Fix, in increasing order of ambition:* (a) pass `sd` **once** via `Pool(initializer=…)` /
a module-global set in the worker initializer instead of per-task args — this alone removes
~50 of the 51 serializations per step; (b) use a `fork` context on Linux/macOS where safe;
(c) slim the facade's `__reduce__` to the ports each worker actually reads. (a) is the
90%-of-the-win, low-risk option. · **1–2 days**

#### A13 · Med-High · robustness · `reconstruction/ecoli/sources.py:24,64-68`; `pyproject.toml:47`

**Two silent upstream-data inputs.**

1. `_DEFAULT_OVERRIDES = Path(__file__).parent / "parca_overrides.tsv"` is applied
   **unconditionally** via `index.update(...)` whenever the file exists (`:64-68`).
   `--bundle-manifest-path` only replaces the *base* manifest; v2ecoli's local overrides always
   win on top, with no CLI flag, no log line, and no record in the output.
2. `from ecoli_sources import BUNDLE_PATH` (`:54-55`) — and `pyproject.toml:47` pins
   `"ecoli-sources"` with **no version**. Every one of the ~140 TSVs the fit reads
   (`knowledge_base_raw.py:20-168,334-340`) comes from a package whose version is unrecorded,
   and no content hash of any TSV is ever computed.

*Failure:* an `ecoli-sources` release, or an edit to `parca_overrides.tsv`, silently changes
the fit with no fingerprint movement anywhere in either layer.

*Fix:* pin `ecoli-sources`; hash the resolved manifest (base + overrides) and the sorted
`(canonical_key, sha256)` list into the provenance sidecar (A11); log the override file when
applied. · **half day**

#### A14 · Med · perf · `v2ecoli/core.py:238-241`

**90 MB of every 266 MB bundle is dead weight on the v2ecoli path.** `simData.cPickle`
(90,169,091 B measured) is written on *every* bundle build, but `core.py:236-237` says
plainly: *"v2ecoli's own composite uses the `configs` dict above and ignores this file."* Only
the vEcoli-comparison path reads it.

*Cost:* **34%** of every build's write I/O, of every per-condition cache's disk footprint, and
of every CI cache upload+download. The `actions/cache` entry rotates on every `sim_data.py`
edit, against a 10 GB repo-wide budget.

*Fix:* `save_sim_input(..., write_raw_simdata=False)` by default; opt in from
`scripts/build_comparison_caches.sh` and the comparison harness. Zero calibration impact — it
is an output, not an input. · **1–2 h**

#### A15 · Med · caching-correctness · `steps/step_03_basal_specs.py:217-221, 227-228, 304-338`

The step-3 Km cache — the *only* intra-fit cache in the entire pipeline — is keyed
`parca-km-{crc32}.cPickle` over `(Km_counts, isEndoRnase, alpha)`.

**What is right:** the un-keyed inputs are re-validated after load (`:311-322`), and the
residual check `np.sum(np.abs(res_aux(Km_cooperative_model))) > 1e-15` means even a CRC
collision is caught and recomputed. This *validate-after-load* design is good and should be
kept.

**What is wrong:** (a) `arrays_differ` is `np.allclose(..., equal_nan=True)` at **default
tolerances** (`:228`) — a cached Km is accepted when the true inputs drifted by up to
`rtol=1e-5`, i.e. a false hit by construction. (b) The key omits the **solver identity** —
not the scipy version, not `method` (unspecified at `:327`, so scipy's default governs), not
`tol=1e-8`, not the code of `km_loss_function`. Given B2, this silently pins the Km fit to
whichever scipy first populated the cache. (c) No atomic write or locking (`:337-338`); two
runs sharing `--cache-dir` — the documented usage at `cli/parca.py:17-18` — can interleave, and
`pickle.load` at `:309` is unguarded. (d) The default `--cache-dir` is `<outdir>/cache` with
`--outdir` defaulting to `out/sim_data`, so the common fresh-output-dir-per-run workflow gets
100% misses and the cache never pays for itself.

*Fix:* SHA256 over all six arrays + dtype/shape + scipy version + solver params; exact
comparison instead of `allclose`; temp-file + `os.replace`; guard the `pickle.load`. · **2–3 h**

#### A16 · Med · perf · `v2ecoli/cli/parca.py:118-146`

**Checkpointing is quadratic and writes multiple GB.** The wrapper accumulates every step's
output into one `running_checkpoint` dict and **re-pickles the whole cumulative state after
every step** (`:132-137`). The final state is **664,887,631 B** uncompressed, so checkpoint 9
alone dumps ~634 MiB and the nine together write several GB — on a 153 s run. This cost is
also **excluded from `runtimes.json`** (the timer stops at `:126`, before the dump at `:133`),
which is precisely why Σ(runtimes.json) understates real wall time.

Secondary: `_wrap` monkeypatches the **Step classes** (`:142`) permanently and process-wide
with no restore, so a second in-process `main()` double-wraps.

*Fix:* write per-step **deltas** (`checkpoint_step_N.pkl` = that step's `out` only) and replay
on resume; or gate behind `--checkpoint`. Restore `cls.update` in a `finally`. · **2–3 h**

#### A17 · Med · perf · `build_ode.py:12-32`

**The step-5 inner ODE has been running interpreted for the life of this code.**
`step_05:176` passes `jit=False`, which selects `self._rates[0]` over `self._rates[1]`
(`equilibrium.py:400-413,431-436`). But `build_functions` **returns the same plain-Python
function twice**:

```python
    f = local_dict["f"]

    return f, f
```

with the comment at `:12-17`: *"This USED TO return a second function that had Numba set up to
JIT-compile the code on demand, but the compiled code time savings don't pay back the
compilation time in Python 3.9+."* So the entire `jit` parameter is dead API surface, and the
equilibrium ODE right-hand side that `solve_ivp` LSODA/BDF calls thousands of times per
condition (`equilibrium.py:441-452`) is `exec`'d interpreted Python (`build_ode.py:27-30`).

*Fix (two independent items):* (i) delete the dead `jit` parameter and the `_rates[0]`/`_rates[1]`
duplication — pure cleanup, zero numerical effect; (ii) *separately*, revisit whether a numba
or `scipy.LowLevelCallable` RHS now pays back, given step 5 is 52 s and the RHS dominates it.
Item (ii) is perf work, not a correctness fix, and should be measured before being attempted. · **(i) 1 h / (ii) 2–3 days**

#### A18 · Med · robustness · `v2ecoli/cli/parca.py:49`; `steps/step_02_input_adjustments.py:195-202`

**`--mode` defaults to `fast`, and fast mode picks its TF by dict insertion order.**

```python
        if self.config.get('debug', False):
            print("  Step 2: debug mode — reducing tf_to_active_inactive_conditions"
                  " to a single key")
            first_key = next(iter(tf_cond))
            tf_cond_out = {first_key: tf_cond[first_key]}
```

Fast mode keeps **whichever TF happens to be first** in the KB's dict — sourced from
`condition/tf_condition.tsv` row order. Reproducible today, but it changes silently if the TSV
is ever reordered, with no `sorted()` and no named TF in the output.

Meanwhile the CLI defaults to `fast` (`:49`) — the mode that mis-calibrates regulation and must
not be simulated — while `v2ecoli/composites/parca.py:68-72` defaults `debug=False`. The two
entry points disagree, and the CLI help (`:50-52`) frames `fast` as merely faster rather than
*wrong*. Fast mode is also what triggers both silent-degradation paths (A3's mechanistic-fit
failures and B1's NNLS fallback, per `metabolism.py:1766-1771`).

*Fix:* default to `full` (it is 2.5 min — the disincentive is imaginary, see A19); name the
selected TF explicitly and `sorted()` it; reword the help to say fast produces a fit that must
not be simulated; stamp `mode` into provenance (A11). · **~1 h**

#### A19 · Med · docs · eight files

**Every documented ParCa runtime is stale by 1–2 orders of magnitude.** Measured: **153.5 s**
(`models/parca/runtimes.json`) and **142.4 s** (`workspace/studies/showcase-1-parca/study.yaml:99`,
Mac mini, `--mode full --cpus 8`, which states outright: *"the docs' '4-8 hours / ~300
conditions' figure is stale … 51 TF conditions fitted"*). A third run in
`docs/parca_workflow_report.html` records 66.6 s.

Stale claims: `docs/generate_full_parca.md:16,37-39` ("4–8 hours", "~300 conditions", "Steps
1–4 take ~10 min"); `models/parca/README.md:14,41-42` ("71.6 min", "the 70-minute step 5");
`README.md:42,492,496` ("~70 min"); `docs/first-run-agent-guide.md:122`;
`step_05_fit_condition.py:3,356` ("~60–70 min at cpus=2"); `cli/parca.py:11-14,50-57,71-72`
("~30 min", "several hours", "~9 conditions each ~6 min"); `data_loader.py:4`;
`scripts/build_cache.py:19`; `scripts/parca_compare.py:143`. Only
`scripts/build_comparison_caches.sh:36` ("~2.5 min") is correct.

*Why this is not cosmetic:* **the stale runtime is the causal reason the fixture is
unreproducible.** Believing a regeneration costs 4–8 hours is exactly why `58353939`
hand-patched a pickle instead of re-running a 2.5-minute pipeline. Fixing the docs unblocks
A11 and A18. · **~1 h**

#### A20 · Med · perf · `step_04:157-163`; `step_08:137`; `step_06`; `composite.py:29-34`

**Three embarrassingly-parallel loops with no parallelism, and `cpus` wired to only two steps.**

- **Step 8** loops over all **51 conditions serially** (`step_08_set_conditions.py:137`) and
  `cpus` is never wired to it at all (`composite.py` passes `cpus` only to steps 4 and 5, `:143,148`).
- **Step 4's combined conditions** (with_aa, acetate, succinate, no_oxygen) run **serially in the
  parent** at `:157-163`, *outside* the pool at `:144-145` — 4–5 full `expressionConverge` runs
  with zero parallelism. (This one is *necessary*: worker mutations to
  `sim_data.process.transcription` don't cross the spawn boundary, which is why results are
  re-applied by hand at `:148-154`.)
- **Step 6** (20.8 s) is a serial fixed point of up to 100 CVXPY solves (`promoter_fitting.py:475`).
- Step ordering itself is forced by artificial tick tokens (`composite.py:29-34`), so steps 6/7
  and 8/9 cannot overlap even where their data allows it.

*Fix:* wire `cpus` into step 8 and parallelize its condition loop via the existing
`apply_updates` (it is a pure per-condition map). Steps 4-combined and 6 are harder and should
wait. · **half day for step 8**

#### A21 · Med · robustness · `v2ecoli/core.py:140-141`; `v2ecoli/cache.py:140-146`

`dill.load(f)` and `json.load(f)` with no `try/except`, no size check, no checksum. A truncated
`sim_data_cache.dill` (see A5) surfaces as a bare `EOFError`/`UnpicklingError` from inside
`dill`, naming neither the cache directory nor the rebuild command — exactly the obscure
traceback `cache_version.py:1-15` exists to eliminate.

*Fix:* wrap and re-raise as `StaleCacheError` reusing `_rebuild_message`. Store a SHA256 of
`sim_data_cache.dill` in `cache_version.json` and check it (~0.3 s for 165 MB — cheap against
the dill load). · **~1 h**

#### A22 · Med · perf · `step_04:218-223,286-291`; `step_08:143-147`; `step_05:152`

**Repeated recomputation of shared per-media quantities.**
`concentrations_based_on_nutrients(media_id=…)` (`metabolism.py:3282-3302`) and
`getBiomassAsConcentrations(doubling_time)` (`growth_rate_dependent_parameters.py:311-336`) are
recomputed from scratch per TF condition (step 4, `:218-223`), per combined condition (step 4,
`:286-291`), and again for all 51 conditions in step 8 (`:143-147`). Neither is memoized, and
many conditions share the same `nutrients` string — step 5 itself relies on that fact when it
dedupes `translation_supply_rate` by nutrients (`:416-419`).

Separately, a fresh `StochasticSystem` is constructed **per seed** (`step_05:152`) — 10
constructions per condition, each re-ingesting the transposed complexation matrix — where one
per condition with a reseeded RNG would do.

*Fix:* memoize both by `(media_id)` / `(doubling_time)`. **Caveat:** memoizing a function that
returns a mutable dict is only safe if callers don't mutate the result — `step_04:222` does
`concDict.update(...)`, so the memo must return a copy. Get that wrong and it *does* change the
fit; hence "safe" here is conditional on returning copies. · **half day**

#### A23 · Med · robustness · `scripts/build_cache.py:44-45`

`os.chdir(repo_root)` where `repo_root` derives from the *script's* location. Combined with
the worktree convention of symlinking `out` → the canonical checkout
(`~/code/v2ecoli--serve-latest/out` does exactly this), running `build_cache.py` from a worktree
writes into the **shared canonical checkout's cache**, overwritten with a fingerprint computed
from the *worktree's* different source tree. The next session in `~/code/v2ecoli` then gets a
`StaleCacheError` it did not cause, or a silently mismatched bundle. The gotcha is
**undocumented**: `grep -i "ln -s"` across `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `docs/`,
`.github/` returns nothing.

*Fix:* refuse to write through a symlinked `out/` unless `--force`; resolve the realpath and
warn. Document the convention in `AGENTS.md`. · **~1 h**

#### A24 · Low-Med · caching-correctness · `scripts/run_condition_multigen_parquet.py:139-155`

The run-manifest "cache fingerprint" is `(st_size, st_mtime_ns)` — **not content** (test at
`tests/test_run_config_cache_fingerprint.py:27-48`). Any `cp -p`, `rsync -a`, `tar -x`, or CI
cache restore preserves mtime, so a restored cache and a rebuilt one are indistinguishable;
conversely a rebuilt-but-identical cache reads as changed. The manifest therefore records a
fingerprint that neither proves identity nor detects change.

*Fix:* reuse `cache_version.json`'s already-content-addressed `inputs_hash`. · **~1 h**

#### A25 · Low · determinism · `getter_functions.py:259`; `cli/parca.py`

**One genuine `PYTHONHASHSEED` dependence, plus an asymmetry.**

```python
                # Add nonduplicate evidence codes to original list
                evidence_list.extend(list(set(new_evidence) - set(evidence_list)))
```

`list(set_difference)` on strings — order varies with `PYTHONHASHSEED` — and it **mutates
`raw_data.transcription_units[...]["evidence"]` in place**. It is mitigated downstream by
`transcription.py:914` (`sorted(tu["evidence"])`), so the observable `rna_id_to_evidence_codes`
is order-stable today; but the raw KB is left in a hash-order-dependent state and any future
unsorted consumer inherits the nondeterminism.

Asymmetry: `PYTHONHASHSEED="0"` is pinned for the **upstream vEcoli** comparison runs
(`vecoli_pbg_upstream.py:455`, `scripts/build_upstream_parca.py:128`,
`scripts/attribute_divergence.py:143`) but **never** by `v2ecoli/cli/parca.py`. The v2ecoli fit
and its reference run under different hash regimes.

*Fix:* `sorted(set(new_evidence) - set(evidence_list))`; set `PYTHONHASHSEED=0` in the ParCa
CLI. *(Sorting a currently-arbitrary order can in principle change the raw KB's evidence order —
but the only consumer already sorts, so the fit output should be bit-identical. Verify.)* · **~1 h**

#### A26 · Low · robustness · `data_loader.py:58-69`; `_scipy_compat.py:41-42,51-52`

Both swallow **all** exceptions on their setup paths (`except Exception: pass` per module; a
silent `return` if the scipy probe fails). A Cython extension that wasn't built
(`scripts/parca_cython_build.sh`) becomes a much later, much more confusing `find_class`
failure inside `_RemappingUnpickler`; a scipy internals change disengages the interpolator
bridge with no signal.

*Fix:* collect and surface failures once at WARNING level, naming the modules; have `install()`
return a bool and log when it self-disables. · **~1 h**

#### A27 · Low · robustness · `v2ecoli/cli/parca.py:120-131` vs `:178-186`

The wrapper writes live-measured per-step times to `runtimes.json` during the run; after the
run the CLI **overwrites the same file** with values regex-scraped from captured stdout
(`re.finditer(r'Step (\d) .*? completed in ([0-9.]+)s')`). Two different definitions silently
swap (the live ones include checkpoint-pickle time, the scraped ones don't — see A16). If any
step's print format drifts from `"  Step N (name) completed in X.Xs"` (emitted at
`step_01:189` … `step_09:171`), the regex yields `{}` and `runtimes.json` is clobbered to empty.

*Fix:* keep the live measurement; use the scrape only to fill gaps. · **~30 min**

#### A28 · Low · robustness · `tests/fixtures/cache/`; `AGENTS.md:138`; `CONTRIBUTING.md:23`

The committed test-fixture cache records `models/parca/parca_state.pkl.gz` = `5701c5d3…`; the
actual shipped fixture hashes to `a46ca8b6…` (verified). It lists only 7 files, including the
long-dead `v2ecoli/composites/baseline.py`. And **nothing in the repo references it** —
`grep -rn "fixtures/cache"` over `*.py` returns zero hits — yet `AGENTS.md:138` says "CI uses a
frozen gzipped cache at `tests/fixtures/cache/`."

*Fix:* delete it (10.4 MB) and fix the two docs, or wire it up and regenerate. Either way,
stop claiming CI uses it. · **~30 min**

**Also noted, latent (not currently reachable):** `fitting.py:1069-1083` extends `rna_indexes`
and `rna_fcs` per perturbed cistron; if two perturbed cistrons share a TU, `rna_indexes`
contains a duplicate. `apply_fcs_to_expression` (`:1035-1058`) dedups via a boolean mask but
`fcs` does not, so the shapes diverge and `scaleTheRestBy` (`:1052`) double-counts. Only
reachable with multi-cistron genotype perturbations. Worth a guard.

---

### Tier B — can change fit output (validate before merge)

#### B1 · High · determinism · `wholecell/utils/fast_nonnegative_least_squares.py:105`; `transcription.py:1952-1955`

**The `set_ppgpp_expression` NNLS — the known non-reproducible step.** `step_03:175` calls
`sd.process.transcription.set_ppgpp_expression(sd)`, which at `transcription.py:1952-1955` does
`self.exp_ppgpp, _ = self.fit_rna_expression(cistron_exp_ppgpp)` →
`fast_nnls(self.cistron_tu_mapping_matrix, cistron_expression)` (`transcription.py:1021-1027`)
→ `nnls(submatrix, b[row_indexes])` at `fast_nonnegative_least_squares.py:105`.

The graph decomposition *around* it is deterministic (the `set()`s at `:40-41` are membership-only;
the driver is `for column_index in range(A.shape[1])` at `:69`). The nondeterminism is
`scipy.optimize.nnls` **itself**: an active-set Lawson–Hanson method, rewritten from Fortran
`nnls.f` to a pure-Python implementation in scipy 1.12. On a degenerate / rank-deficient
submatrix — **very common for polycistronic TU mapping**, where many cistrons map to one TU —
the active set chosen at a tie can differ across scipy versions and LAPACK backends.
`pyproject.toml:39` pins only `scipy<1.18` with **no floor**, so 1.11 and 1.17 are both permitted.

*Fix:* pin a scipy floor; add a deterministic tie-break (e.g. lexicographic on column index
among equal-residual active sets); assert and record the residual so drift is observable.
*Risk:* **changes fit output** if the current run lands on the other tie branch. Bit-compare. · **1–2 days**

#### B2 · High · determinism · `metabolism.py:1840-1873`

**A second, distinct NNLS — the Step-9 amino-acid kcat fallback.**
`nnls(A_mat, b_vec, maxiter=50)` on a **2×2** that is degenerate *by construction*: the comment
at `:1776-1779` says the expected answer has one kcat pinned at **zero**, i.e. a boundary
solution on an active-set method. Which kcat goes to zero can flip across scipy versions and
BLAS backends. `kcat_fwd` is written straight to `data["kcat"]` (`:1853`) and the recomputed
`fwd_rate`/`rev_rate`/`deg_rate` (`:1861-1872`) propagate into downstream amino acids — a flip
is a different metabolic parameterization, not a rounding difference. And the comment at
`:1766-1771` says the fallback fires *more often* under fast mode, coupling it to A18.

*Fix:* a 2×2 non-negative least-squares has a closed-form solution set — enumerate (interior
solve, plus the two boundary solves) and take the lowest residual with **deterministic
tie-breaking**. Log which branch fired. *Risk:* changes fit output where the branch flips.
Bit-compare. · **1 day**

#### B3 · High · determinism · `promoter_fitting.py:495,540`; `pyproject.toml:42`

**Step 6 is the largest reproducibility exposure and it is entirely unpinned.**

```python
        prob_r.solve(solver="ECOS", max_iters=1000)     # :495
        prob_p.solve(solver="ECOS")                     # :540
```

An interior-point conic solver on a 1-norm objective (`PROMOTER_NORM_TYPE = 1`, `:30`), inside
a fixed point of up to 100 iterations (`PROMOTER_MAX_ITERATIONS`, `:31`), convergence tolerance
`1e-9` (`:32`), with post-hoc snapping at `ECOS_0_TOLERANCE = 1e-10` (`:33,506,551-552`).
`cvxpy` (`pyproject.toml:42`) and `ecos` are both **unpinned**. Different canonicalization or
ECOS build → different `r`/`p` at the snap boundary → different `pPromoterBound` →
different `basal_prob`/`delta_prob`. Note `prob_r` sets `max_iters=1000` and `prob_p` uses the
default — asymmetric and undocumented. Compounding: `promoter_fitting.py:346-361` builds
`pAlphaIdxs`/`pNotAlphaIdxs`/`fixedTFIdxs` by iterating `H_col_name_to_index.items()` in dict
order, and those index arrays determine the constraint-matrix layout ECOS sees.

*Fix:* pin `cvxpy` and `ecos` to exact versions; set `max_iters` on both solves; `sorted()` the
index-array construction; record the achieved objective per outer iteration. *Risk:* pinning
alone should be a no-op if the current env already matches; sorting the index arrays can change
the ECOS path. Bit-compare. · **1 day**

#### B4 · Med · determinism · `step_03_basal_specs.py:275,292,327`; `transcription.py:225,2031`; `trna_charging.py:187`

**Unpinned optimizer/solver defaults across the fit.**
Three `scipy.optimize.minimize(loss, …, jac=loss_jac)` calls with **no `method=`** (`:275,292`
also with no `tol`), so a scipy default change silently changes the algorithm.
`np.linalg.lstsq(A, fraction_active, rcond=None)` at `transcription.py:225` (LAPACK `gelsd`,
backend-sensitive). `Flst = np.linalg.inv(F.T.dot(F)).dot(F.T)` at `transcription.py:2031` — an
explicit normal-equation inverse (κ² conditioning) where `lstsq` belongs.
`solve_ivp(dcdt, …, method="BDF")` at `trna_charging.py:187` with **no `rtol`/`atol`**.
`equilibrium.py:441-450` loops `for method in ["LSODA", "BDF"]` — so **which integrator runs
depends on whether LSODA raised**, i.e. on the data.

Related, already partially handled: `transcription.py:2093-2107` carries a **hardcoded
`1e-10 * old_prob.max()` floor** with a comment stating the blow-up "depends on the BLAS/LAPACK
build, which made exp_free / exp_ppgpp non-reproducible across numpy/scipy versions." That is
the right instinct, but the threshold is an untracked magic number whose validity rests on an
empirical data property (dust band ≤1e-13 vs real genes ≥1e-9) that nothing enforces.

*Fix:* set `method=` and explicit `tol`/`rtol`/`atol` on every solver; replace the normal-equation
inverse with `lstsq`; assert the 1e-10 floor's separation assumption. *Risk:* changes fit
output (different optimizer paths). Bit-compare. · **half day**

#### B5 · Med · perf / caching · `composite.py:232-305`

**Layer 1 is all-or-nothing.** The only granularity is the manual
`--resume-from-step N --resume-pickle …` escape hatch (`cli/parca.py:69-74`) and the step-3 Km
cache. Steps 4 and 5 recompute everything on every run; there is no per-condition and no
per-step content-addressed artifact store.

*Estimated win:* from the measured profile, content-addressed per-step caching makes
step-9-only iteration **~20×** faster (6.8 s vs 153.5 s) and step-6-onward **~5×** faster
(27.7 s vs 153.5 s). The infrastructure is *almost* there — `checkpoint_step_N.pkl` already
exists, it just isn't keyed by input content (and A16 should land first so the checkpoints
aren't cumulative).

*Fix:* `key = SHA256(step source + config + hash of the upstream store slice)`; reuse the
checkpoints as the artifact store; skip on hit. *Risk:* **low but nonzero** — a skipped step
must be provably a pure function of its declared inputs, and steps 4/5 reach `sim_data` through
`make_sim_data_facade`, so the input slice needs care. Bit-compare cold vs warm. · **3–5 days**

#### B6 · Med · determinism · `fitting.py:951-1001`; `parallelization.py:62-63,201-208`

**Does `--cpus` change the fit? Structurally no; numerically, not guaranteed.**

*Structurally clean — verified.* No chunked reductions, no cross-worker float summation. Each
worker returns a whole per-condition dict; the merge is `dest.update(...)` of disjoint keys in
a fixed order (`results` is insertion-ordered by `labels`, and both callers pass `sorted`
labels — `step_04:133`, `step_05:383`). `grep` for `sim_data.<attr> =` and
`sim_data.<attr>[…] =` across `fitting.py`, `step_04`, and `step_05` returns **nothing** — the
workers are pure w.r.t. `sim_data`.

*But three mechanisms can still make it matter at the last bits:*
1. **BLAS thread contention.** Nothing on the ParCa path sets `OMP_NUM_THREADS` /
   `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`. With `--cpus 8` on an 8-core box, each spawned
   worker's OpenBLAS may pick a different thread count than the serial run, changing GEMM
   reduction order. That feeds a fixed point with `FITNESS_THRESHOLD = 1e-9` /
   `MAX_FITTING_ITERATIONS = 200` (`fitting.py:270-271`), so the iteration *count* at which
   convergence trips can differ, and the returned `expression` with it.
2. **`cpus()` reads SLURM env** (`parallelization.py:62-63`), so the effective worker count
   depends on scheduler variables nobody records (A8's sibling).
3. **The `stochasticRound` retry loop** (`equilibrium.py:477-483`) converts any such ulp
   difference into an **integer molecule-count** difference.

*Also:* `--cpus` defaults to `os.cpu_count()` (`cli/parca.py:32-33,53`) — **machine-dependent by
default**. And step 5 passes the raw config value (`:400`) where step 4 sanitizes through
`parallelization.cpus()` (`:131`) — on SLURM, step 5 can over-subscribe. Failure semantics
differ too: `cpus>1` collects per-label exceptions into a combined `RuntimeError`
(`fitting.py:980-996`); `cpus==1` propagates the first immediately.

*Fix:* pin BLAS thread env vars to 1 inside ParCa workers; route step 5 through
`parallelization.cpus()`; record the resolved cpu count in provenance (A11). **Add a CI test**
that runs the pipeline at `cpus=1` and `cpus=2` and asserts bit-identical `parca_state`.
*Risk:* pinning BLAS threads **will** change numbers on machines that were previously
multi-threading. That is the correct trade (determinism over a small speed loss), but it must
be bit-compared. · **1 day (test first, it de-risks everything else)**

---

## What is already right (don't regress it)

Worth stating explicitly, because a rewrite could easily lose these:

- **Deterministic seeding.** `step_05:139,152,164` — `for seed in range(N_SEEDS)` with
  `StochasticSystem(random_seed=seed)` and `RandomState(seed)`. No wall-clock or PID seeding
  anywhere in the fit.
- **Deterministic iteration order where it counts.** `sorted(set(...) | ...)` at
  `step_05:104-111`; `sorted(sd.tf_to_active_inactive_conditions)` at `step_04:133`;
  `sorted(concDict)` at `step_05:100`; `sorted(set(tf_list))` at `relation.py:133` (load-bearing —
  it drives `G`/`H` column order in `promoter_fitting.py`). No `os.listdir`/`glob`/`os.walk`
  anywhere under `processes/parca/` — flat files come from the explicit ordered
  `LIST_OF_DICT_FILENAMES` (`knowledge_base_raw.py:20-168`).
- **No `deepcopy` in the ParCa path at all.** Copies are shallow and explicit (`dict(...)` on
  `cell_specs`, `.copy()` on arrays). This is a genuine strength.
- **Pure worker functions.** `buildTfConditionCellSpecifications` (`step_04:193-255`) returns a
  dict; the caller re-applies at `:148-154`. This is what keeps B6 structurally clean.
- **`KnowledgeBaseEcoli` is constructed exactly once per run** (`cli/parca.py:97-101`). No
  redundant flat-file reload.
- **Repo-anchored fingerprint root.** `cache_version._default_repo_root` (`:102-113`) resolves
  from `__file__`, not `cwd`, with an excellent comment about the `os.chdir`-into-`.regen_*`
  failure it prevents. Sound reasoning — A1 defeats it by a different route, not by breaking it.
- **Marker written last.** `cache_version.json` at `core.py:255`, after the dill at `:229`.
  Right order; A5 is about rebuild-over-existing only.
- **Validate-after-load on the Km cache** (`step_03:311-322`). The *idea* is exactly right; A15
  is only about the tolerance, the partial key, and atomicity.
- **The reference implementation is already in-repo.**
  `v2ecoli/comparison/vecoli_parca.py:45-60,167-176` fingerprints the **output** — *"a
  fingerprint of the result, not of the inputs, so a nondeterministic refit at the same address
  is detectable"* — and records a `context` block with python/scipy/numpy. Its docstring
  (`:12-19`) diagnoses the exact scipy-drift failure Layer 2 is still exposed to. A9 is
  "adopt what the module next door already proved."

---

## Verification plan

The repo already has the right levers.

**Calibration-sensitive artifacts:**
- `tests/golden/polypeptide_elongation_baseline.json` — regenerated in `71442761` when the
  fixture changed, so it demonstrably moves with the fit.
- `tests/golden/baseline_parity_signature.json`.
- `tests/compare/test_parca_drift.py` — field-by-field diff of the hydrated
  `SimulationDataEcoli` against vEcoli's across all named conditions. Currently opt-in (skips
  unless both fits are present). **Make it a required gate for every Tier B change**, with
  `V2_PARCA_STATE` pointed at the freshly built state.

**Procedure for every Tier B change:**

1. **Baseline capture.** One machine, one venv, fixed env
   (`V2PARCA_N_SEEDS` unset, `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=1`):
   ```
   v2ecoli-parca --mode full --cpus 8 -o out/parca_base
   shasum -a 256 out/parca_base/parca_state.pkl
   python scripts/build_cache.py --fixture … --cache out/cache_base
   shasum -a 256 out/cache_base/sim_data_cache.dill
   ```
2. **Apply the change, rebuild identically** into `out/parca_new`.
3. **Bit-compare first.** If `sha256(parca_state.pkl)` is unchanged, the change is
   calibration-neutral and needs nothing further. **This is the pass criterion for every Tier A
   item** — A1–A28 should all produce a byte-identical `parca_state.pkl` (A22 and A25 are the
   two to watch: A22 only if the memo returns copies, A25 only if the sorted evidence order is
   genuinely unobserved downstream).
4. **If bytes differ, diff numerically.** `pytest tests/compare/test_parca_drift.py` with
   `V2_PARCA_STATE=out/parca_new/parca_state.pkl`, plus `scripts/parca_compare.py` for the
   field-level report. Every differing field must be explainable by the change; unexplained
   drift blocks the merge.
5. **Behavior gate.** Rebuild `out/cache` from the new state; run `pytest -m "sim and not slow"`
   and both `test_polypeptide_elongation_parity` tests. A golden regeneration must be a
   **separate, explicitly-justified commit** (`AGENTS.md`: "Do not modify anything under
   `tests/fixtures/`").
6. **Cross-machine determinism (B1–B4, B6).** Run the full fit on this Mac *and* on the mini and
   compare `sha256(parca_state.pkl)`. **If they differ today, that difference *is* the bug** —
   and the fix's success criterion is that they stop differing.

**For A1/A2/A4** (the fingerprint repair) verification is different and easier — these change
the *key*, not the fit:
- `sha256(sim_data_cache.dill)` before and after must be **identical** (content must not move).
- `inputs_hash` **must** move (that is the point), and `SCHEMA_VERSION` must be bumped.
- The two new guards (`test_input_files_all_exist`, `test_ci_key_matches_input_files`) must
  fail on `main` and pass after.

---

## Suggested sequencing

**Week 1 — restore the guarantees (all Tier A, all calibration-neutral).**
A1 → A2 → A4 (fingerprint repair + guard tests) → A3 + A6 (stop the two silent-degradation
paths) → A5 + A21 (corruption + messaging) → A19 (docs; unblocks A11/A18) →
A10 + A14 (two free perf wins: memoize `stoich_matrix`, drop the 90 MB write) →
A18, A23, A26, A27, A28 (cheap).

**Week 2 — close the untracked-input gaps.**
A8 (`V2PARCA_N_SEEDS`) + A9 (dependency context, incl. pinning cvxpy/ecos/scipy-floor) +
A7 (condition/media/seed in the key) + A13 (`ecoli-sources` + overrides) + A15 (Km key) +
A11 (provenance sidecar) + A16 (checkpoint deltas), then **one clean regeneration** of
`models/parca/parca_state.pkl.gz` with `manual_patches: []` — retiring the `58353939`
hand-patch and making the calibration reproducible for the first time.

**Week 3+ — perf, gated by the verification plan.**
A12 (pass `sd` via pool initializer — the single biggest perf item) → A20 (parallelize step 8)
→ A22 (memoize per-media, with copies) → A17(i) (delete the dead `jit` branch).

**Then, calibration-risky, in this order.**
B6's **test first** (cpus invariance — a test, not a change, and it de-risks the rest) →
B6's BLAS pinning → B4 → B3 → B1 → B2 → B5.

If only one thing ships from this review, ship **A1 + A2**: they convert the cache guard from
decorative back into load-bearing. If two, add **A3 + A6**: they stop the pipeline from
producing a silently-degraded calibration that looks identical to a good one.
