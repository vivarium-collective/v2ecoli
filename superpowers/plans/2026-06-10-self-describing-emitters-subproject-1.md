# Self-Describing Emitters (Readout-Coordination Sub-project #1) — Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make v2ecoli runs **self-describing** — persist listener-vector element NAMES with the data so the evaluator can resolve readouts/aggregates (e.g. "sum across DnaA forms", `monomer_counts`) from a stored run alone, no `sim_data`. Mirror vEcoli's mechanism: annotate listener `outputs()` port schemas with element names, harvest them via an `output_metadata()` walker, and feed them into the emitter config (xarray `id` coord = names; parquet `output_metadata__<field>`; sqlite catalog).

**Architecture / key facts (grounded):**
- The emitters ALREADY persist a provided catalog — pbg-emitters xarray fills an `id_<var>` coordinate (currently with integer `0..N-1`, see `pbg_emitters/xarray_emitter/storage.py` `VAR_COO_PREFIX="id_"`), parquet writes `output_metadata__<field>` from `config["metadata"]` (`pbg_emitters/parquet_emitter.py` `METADATA_PREFIX`, `_write_configuration`), sqlite has a `simulations.metadata` slot. So the WRITE plumbing exists; **the missing piece is the source of names**, which must come from v2ecoli (the emitter only sees values; emit types are erased to `node`).
- vEcoli pattern to mirror: `vEcoli/ecoli/library/schema.py:596` `listener_schema` (a `(default, names)` tuple → `_properties.metadata`); `vEcoli/ecoli/experiments/ecoli_master_sim.py:1011` `output_metadata()` (iterate processes, `get_schema`, pull `_properties.metadata` leaves, `inverse_topology` re-root, deep-merge); `:852` `metadata["output_metadata"] = output_metadata()`.
- v2ecoli today: listeners declared bare (`v2ecoli/composites/_helpers.py` `'monomer_counts': 'array[integer]'`; `v2ecoli/processes/polypeptide_initiation.py` `ribosome_init_event_per_monomer`); element names live in process configs (`v2ecoli/library/sim_data.py` `monomer_ids` ~:1073/:1780). A vivarium-style listener_schema helper is vendored in `v2ecoli/library/schema.py` (supports the metadata convention) but unused with the metadata form. `v2ecoli/library/xarray_run.py` `extract_output_metadata_from_state` currently emits `range(N)` integer coords — to be replaced/augmented with real names.

**Tech Stack:** Python 3.11+ (run via `.venv/bin/python` — bare python lacks `unum`); process-bigraph; pbg-emitters; pytest. Spec: `pbg-superpowers/docs/specs/2026-06-09-readout-coordination-design.md` (sub-project #1).

**Repo:** v2ecoli (branch `feat/self-describing-emitters`, off `main`). Light verification in pbg-emitters (separate, optional).

**Scope guardrails:**
- Backward-compatible: the `(default, names)`/`_properties.metadata` annotation must NOT change runtime behavior of existing listeners (the vendored helper supports it; process-bigraph treats `_properties` as an annotation). Run the relevant v2ecoli tests/a tiny sim to confirm nothing breaks.
- Start with the listeners the dnaa evaluator needs: `monomer_counts`, `rnap_data.*` (init-rate), `replication_data.*` (oriC). Don't annotate everything.
- Golden proves names reach the emitter CONFIG (no full ParCa rebuild). A tiny sim only if cheap using the existing cache.

---

## File map (verify exact paths against live code first)
- `v2ecoli/library/schema.py` — confirm/extend the vendored `listener_schema` helper for the `(default, names)` convention.
- `v2ecoli/processes/*.py` and/or `v2ecoli/composites/_helpers.py` — annotate the target listener `outputs()` with names from their config.
- `v2ecoli/library/output_metadata.py` (new) — the `output_metadata()` walker.
- `v2ecoli/library/xarray_run.py` / the emitter-config build path — wire `output_metadata()` into `config["metadata"]["output_metadata"]` (replace the `range(N)` coord source).
- Tests under `v2ecoli/.../tests/`.

---

## Task 1: `output_metadata()` walker

**Files:** Create `v2ecoli/library/output_metadata.py`; Test alongside.

- [ ] **Step 1: Failing test** — build the baseline composite (use `v2ecoli.core.build_core()` + the baseline generator; reuse how the dashboard/tests build it), call `output_metadata(composite)`, assert it returns `{}` initially (no annotations yet) — establishing the walker contract. (After Task 2 it returns names.)
- [ ] **Step 2: Run → fail** (`.venv/bin/python -m pytest <test> -v`).
- [ ] **Step 3: Implement** `output_metadata(composite_or_core) -> dict[str, list]` mirroring vEcoli `ecoli_master_sim.py:1011`: iterate process/step instances in the composite, get each one's output port schema (`get_schema`/`outputs()`), pull leaves carrying `_properties.metadata`, re-root the port-relative path to the absolute store path (use the composite topology / an inverse-topology helper), deep-merge into one dict. Read vEcoli's `extract_metadata` (`ecoli_master_sim.py:1055`) + `inverse_topology` for the exact shape.
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(output_metadata): walker harvesting _properties.metadata from listener schemas`

## Task 2: Annotate the dnaa-relevant listener outputs with names

**Files:** `v2ecoli/library/schema.py` (helper, if needed) + the target listener process `outputs()`.

- [ ] **Step 1: Failing test** — for the annotated listener (start with `monomer_counts`), assert its `outputs()`/`get_schema` carries `_properties.metadata` = the element names (from the process config, e.g. `monomer_ids`), AND `output_metadata(composite)` (Task 1) now returns `{<store_path>: [names...]}` for it.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — use the vendored `listener_schema((default, names))` convention (confirm it produces `_properties: {metadata: names}` and is realized without error by process-bigraph). Annotate `monomer_counts` (names = `monomer_ids` from config), then `rnap_data` init vectors and `replication_data` as needed. Names come from the producing process's config (already available at construction). DO NOT change the listener's runtime values — only add the metadata annotation.
- [ ] **Step 4: Run → pass; AND run a representative existing listener/sim test to confirm no behavior change.** **Step 5: Commit** — `feat(listeners): annotate monomer_counts/rnap/replication outputs with element names`

## Task 3: Wire `output_metadata()` into the emitter config at run build

**Files:** the v2ecoli run-build path (`xarray_run.py` and/or the composite-generator emitter config / wherever `config["metadata"]`/emitter is assembled per run).

- [ ] **Step 1: Failing test** — building a run's emitter config (the path the dashboard/CLI uses) now includes `config["metadata"]["output_metadata"]` = the walker's output; for xarray, the `id_<var>` coord source is the names (not `range(N)`).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — at emitter-config assembly, call `output_metadata(composite)` and set `config["metadata"]["output_metadata"]` (parquet/sqlite) and the xarray `output_metadata` coord input (so `id_<var>` carries names). Reconcile/replace `extract_output_metadata_from_state`'s `range(N)` default with the names where available; keep `range(N)` fallback for un-annotated vectors (never break — un-named vectors still get an integer coord).
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(run): feed output_metadata names into emitter config (xarray id coord / parquet output_metadata)`

## Task 4: Golden — a run is self-describing (config-level; tiny-sim optional)

**Files:** Test.

- [ ] **Step 1: Config-level golden** — build the baseline composite + emitter config for the default emitter, assert the names for `monomer_counts` (and the others) are present in the config (parquet `output_metadata`) / the xarray coord inputs. This proves names reach the store without a full ParCa rebuild.
- [ ] **Step 2: (Optional, if cheap on the existing cache)** run a minimal sim (1 gen, few steps, existing `out/cache`/`out/kb`) to a tmp out dir; open the resulting store with `pbg_emitters.RunReader` and assert the names are recoverable from the store alone (e.g. parquet `output_metadata__monomer_counts` via `field_metadata`, or xarray `id_monomer_counts` coord = names). NEVER write to the user's `out/` — use a tmp dir.
- [ ] **Step 3: Full suite** `.venv/bin/python -m pytest -q` (the relevant subset) — confirm green; no existing sim/listener test regressed.
- [ ] **Step 4: Commit** — `test(self-describing): golden — listener names reach the run store`

## Task 5 (light, separate): pbg-emitters consumption check
- [ ] Confirm pbg-emitters consumes the catalog across backends (xarray `id` coord already; parquet `output_metadata__` already; **sqlite** — add catalog persistence to `simulations.metadata` + a RunReader read path IF missing). This may be deferred into sub-project #2 (RunReader catalog) — note status. (Do NOT block #1 on sqlite if the default emitter is parquet/xarray.)

---

## Self-Review
- Spec coverage (sub-project #1 = self-describing stores): names persisted via xarray id-coord + parquet output_metadata (+ sqlite noted) → Tasks 2-4; the walker mirroring vEcoli → Task 1; backward-compat + don't-break-sims → Task 2/3 guardrails. #2 (RunReader read/resolve) and #6 (evaluator via readouts) are separate.
- No placeholders: design + vEcoli refs are concrete; the implementer grounds exact v2ecoli code/paths before each task (paths above are to-verify).
- Risk: if process-bigraph does NOT realize `_properties.metadata` cleanly (breaks the listener), STOP and report (BLOCKED) — do not force it; the fallback is the run-wiring lookup (the alternative the user did not pick) as a contingency.

## Notes for the executor
- Branch `feat/self-describing-emitters` (off main) is set; do not switch to the user's `feat/baseline-configure-knobs`.
- `.venv/bin/python -m pytest` (bare python lacks `unum`).
- Ground exact paths/signatures in LIVE v2ecoli + vEcoli (`/Users/eranagmon/code/vEcoli`) before each task — the file:line refs here are from a prior survey and must be verified.
- Use the existing ParCa cache (`out/kb/simData.cPickle`, `out/cache`); NEVER rebuild ParCa, never write to the user's `out/`.
- pbg-emitters RunReader is importable in v2ecoli's `.venv`? Verify; if not, `uv pip install -e /Users/eranagmon/code/pbg-emitters[parquet]` into v2ecoli's venv for the optional sim-golden.
