# `translate_ports` completeness audit

Static + empirical audit of the vivarium-`ports_schema()` → bigraph-schema
converter against the ~32 un-migrated vEcoli processes, so converter gaps can be
fixed proactively rather than discovered one tick at a time.

- **Converter under audit:** `/Users/eranagmon/code/vEcoli-cf/ecoli/library/bigraph_types.py::translate_ports(core, ports, path=())` (lines 581–639) plus its `infer()` dispatches.
- **Bridge that consumes it:** `/Users/eranagmon/code/v2e-compare-harness/v2ecoli/library/vivarium_bridge.py` (`wrap_vivarium_process`, and a *second, divergent* `translate_ports`).
- **Empirical evidence:** `/Users/eranagmon/code/v2e-compare-harness/tests/test_translate_ports_audit.py` (11 cases, all green) feeds real fragments to the real converter. The "handled?" column below is backed by those assertions.
- **Un-migrated set (32):** processes under `ecoli/processes/**` that define `ports_schema()` but not `def inputs(self)`. Saved to `/tmp/unmig.txt`. They are almost all *peripheral* (antibiotics, chemotaxis, environment, spatiality, listeners) — the one core-adjacent one is `metabolism_redux_classic.py` (alt-metabolism, and the source of the `media_id` bug).

> **Two converters, diverging.** The vEcoli-cf `translate_ports` (audited here) does **NOT** special-case `bulk`/unique stores, while the harness copy in `vivarium_bridge.py` does (`_special_type`: `bulk`→`bulk_array`, unique→`unique_array`). Several gaps below are already solved in the harness copy and just need porting back, or the harness copy should be the one `wrap_vivarium_process` uses. Confirm which converter the live wiring path actually calls before fixing.

---

## Top 5 fixes likely to unblock the most processes

1. **Separate input vs output declarations (the `media_id` bug class).** `translate_ports` has no read/write distinction: `{_default:"", _updater:"set"}` renders `overwrite[string]`, and `wrap_vivarium_process` declares the **same** `_typed_ports` for both `inputs()` and `outputs()`. A port a process only *reads* but that carries `_updater:set` (e.g. `environment.media_id`) is then declared `overwrite[string]` on the **input** side, injecting `Overwrite` structure into a shared store where a sibling produces a bare string → "media_id arriving as a dict". **Fix:** `inputs()` should use the plain (Overwrite-stripped) type; `outputs()` should carry `Overwrite` and should be restricted to ports the process actually writes via `wrap_vivarium_process(output_ports=...)`. Bites at realize/wiring time — fix first.

2. **Map `bulk`/unique stores to their registered array types.** The audited converter infers a raw structured-array `Array` for `"bulk": numpy_schema("bulk")` (11 un-migrated processes touch `bulk`), which (a) collides with the core composite's `bulk_array` at the shared store and (b) loses the sparse `[(idx, delta)]` updater. Port the harness's `_special_type` (bulk→`bulk_array`, unique molecules→`unique_array[...]`) into the vEcoli-cf converter / conversion path.

3. **Translate `_divider`.** `_divider` is dropped entirely (test: `binomial_ecoli` → plain `integer`). 43 divider declarations across the set: `zero`(11), `set_value`-dict(10), `set`(7), `split`(5), `binomial_ecoli`(2), `split_dict`, `empty_dict`, and **callables** (`divide_lattice`, `daughter_locations`). At division, daughters get bigraph's default behavior instead of split/binomial/zero. Minimum: map `_divider:"set"` → `Overwrite` (already implied by set updater) and wire `split`/`binomial` to the existing `divide` dispatches.

4. **Interpret inline "update-with-updater" return dicts.** 6 processes (`local_field`, `diffusion_field`, `reaction_diffusion_field`, `lysis`, `chemostat`, `tetracycline_ribosome_equilibrium`) return updates shaped like `{"_value": delta, "_updater": "nonnegative_accumulate"}` from `next_update`. The bridge passes `next_update`'s return straight through as the bigraph update; `apply()` won't interpret the embedded vivarium directive. Unwrap/translate these in `wrap_vivarium_process.update()`.

5. **Don't over-infer partially-typed collections at shared stores.** The empty-collection→`Node` guard only catches *empty* `set()/{}/[]/None`. A *non-empty* dict/list/structured-array default still infers a concrete `Map`/`List`/`Array` that can conflict with a hand-typed sibling at a shared store. Where a port name is a known shared store, prefer `Node`, or rely on fix #1's `output_ports` to keep the converter off the write surface.

---

## Feature table (ordered by when in a tick the gap bites)

| # | Feature / vivarium construct | Used by (un-migrated processes) | Handled today? | Failure mode if unhandled | Suggested handling |
|---|---|---|---|---|---|
| **A. Wiring / realize time (earliest)** |
| A1 | **Bidirectional port declared `_updater:set` but only read** (`overwrite[...]` on input side) | `metabolism_redux_classic` (`environment.media_id`, `next_update_time`); any `set`-updater port (51 `set` decls) | **No** — no read/write split; `Overwrite` applied to both `inputs()`+`outputs()` | Overwrite structure injected at a shared store; reader sees a dict/wrapper where a scalar (string `media_id`) is expected | Strip `Overwrite` in `inputs()`; emit it only in `outputs()`; pass `output_ports=` the written set |
| A2 | **`bulk` / unique structured-array stores** (`numpy_schema("bulk")`, unique molecule schemas) | 11 with `bulk`: `metabolism_redux_classic`, `enzyme_kinetics`, `rna_interference`, `conc_to_counts`, `pbp_binding`, `cell_wall`, `murein_division`, `permeability`, `tetracycline_ribosome_equilibrium`, `lysis`, `exchange_stub` | **No** (vEcoli-cf converter); **Yes** in harness `vivarium_bridge` copy | Raw structured `Array` conflicts with core's `bulk_array` at shared store; sparse `[(idx,delta)]` updates don't apply | Port `_special_type` (bulk→`bulk_array`, unique→`unique_array[...]`) into the audited converter |
| A3 | **Glob `{'*': sub}` (sole child)** | `flagella_motor`, `local_field` (exchanges, fields), `multibody_physics`, `spatial_geometry`, `diffusion_network`, `metabolism_redux_classic` (`boundary.external.*`) | **Yes** → `map` (incl. nested, and with underscore siblings filtered) | — | — (works; keep regression test A3) |
| A4 | **Glob `'*'` mixed with a NAMED sibling at same level** | none currently (latent) | **No** → literal `'*'` store child | Process reads back `{'*': ...}` instead of `{key: value}` | Build `map` from the `'*'` entry and merge fixed named children, or assert-and-warn |
| A5 | **Non-empty dict / list / nested-dict defaults** (concrete `Map`/`List`) | `aggregator` (`{}`), `local_field` (`initial_external`), `metabolism_redux_classic` (`maintenance_reaction`, weights), many | **Partial** — concrete type inferred; only *empty* collections fall back to `Node` | Concrete `Map`/`List` conflicts with a hand-typed sibling at a shared store | Widen the `Node` fallback for known shared stores, or keep converter off the write surface (fix A1) |
| A6 | **`pint.Quantity` defaults** (`0 * units.mM`, `1 * units.um`, `... g/L`) | 18 files (`metabolism_redux_classic`, all `antibiotics/*`, `environment/*`, `derive_globals`, `concentrations_deriver`) | **Yes** — `infer(pint.Quantity)` → `Quantity`; vivarium ureg shared | — | — |
| A7 | **Unum / `UnitStructArray` / `csr_matrix` / sympy defaults** | not in this un-migrated set (core-process territory) | **Yes** — dedicated `infer` dispatches | — | — |
| A8 | **numpy `ndarray` defaults** (`np.ones`, `np.zeros`, 2-D fields) | 18 files (diffusion fields, `cell_wall` lattice, etc.) | **Yes** → `array[shape,dtype]` | Object/ragged-dtype arrays would degrade, but none seen | — |
| A9 | **Callable / bound-method defaults** | none in the un-migrated `ports_schema` defaults (callables appear only as `_divider`/`_updater` values, see C/D) | n/a | — | — |
| **B. Per-tick update logic** |
| B1 | **Inline "update-with-updater" return dicts** `{"_value":…, "_updater":"nonnegative_accumulate"}` | `local_field`, `diffusion_field`, `reaction_diffusion_field`, `lysis`, `chemostat`, `tetracycline_ribosome_equilibrium` | **No** — passed through verbatim; `apply()` can't read the embedded directive | Update silently mis-applied (treated as a literal dict value) | Unwrap in `wrap_vivarium_process.update()`: translate `{_value,_updater}` to the matching `apply` path |
| B2 | **`_updater` as a registry callable** `updater_registry.access("accumulate")` | `engine_process` | **Partial** — not `== "set"`, so falls through to `infer(default)` (accumulate semantics, which is the default) → correct only because it *is* accumulate | A registry callable other than accumulate/set would be silently ignored | Resolve registry callables to their name; map `set`-like ones to `Overwrite` |
| B3 | **`nonnegative_accumulate` updater** | `local_field` | **Partial** — not `set`, infers plain numeric → plain accumulate; loses the non-negativity clamp | Counts can go negative where vivarium clamped to 0 | Map to a clamped-accumulate `apply`, or post-clamp |
| **C. Division time (latest)** |
| C1 | **`_divider` string variants** `zero`/`set`/`split`/`binomial_ecoli`/`split_dict`/`empty_dict` | `conc_to_counts`, `cell_wall`, `pbp_binding`, `murein_division`, `flagella_motor`, `tetracycline_ribosome_equilibrium`, `death`, others (43 decls) | **No** — `_divider` is never read (test: `binomial_ecoli`→`integer`) | At division, daughters use bigraph default instead of split/binomial/zero | Translate `_divider`→registered `divide` behavior; at minimum `set`→`Overwrite` |
| C2 | **`_divider` dict form** `{"divider":"set_value","config":{"value":…}}` | `cell_wall` (×7), `pbp_binding` (×4), `murein_division` | **No** | daughter values not reset to the configured constant | Parse dict divider → set-to-constant divide |
| C3 | **`_divider` callable + topology** `{"divider": daughter_locations, "topology": {...}}`, `divide_lattice` | `multibody_physics`, `cell_wall` | **No** | positional/lattice daughters mis-divided | Mirror the `UniqueArray` divide path (resolve topology against context) for these |
| **D. Cosmetic / no-op** |
| D1 | `_emit` (80 decls), `_properties`, `_serializer` | many | **Ignored (correct)** — emit is output-side, doesn't affect the type tree | — | none needed |

---

## Input-vs-output declaration (the active bug class)

`wrap_vivarium_process` over-declares: with `output_ports=None` (default) **every** port goes into both `inputs()` and `outputs()` from one `_typed_ports` tree. Two consequences:

1. `Overwrite`-typed ports (any `_updater:set`) land on the **input** side, injecting wrapper structure into shared scalar stores (`media_id`). Fix: build inputs from Overwrite-stripped types.
2. Read-only ports land on the **output** side. For accumulate/`Float` this is harmless, but for any structured/typed port it widens the write surface and risks resolve conflicts.

**Determining the write surface.** A port is written iff its name is a top-level key returned by `next_update()`. Note `_updater` presence in the schema is **not** a reliable proxy — vivarium's default updater is `accumulate`, so processes routinely write ports that declare no `_updater` (e.g. `exchange.internal`, see below). So:

- Restricting outputs to "ports with `_updater`" would **under**-declare and drop real accumulate writes.
- Over-declaring outputs is the safe default for *type correctness*; the `Overwrite`-on-input problem (1) is the real bug and is fixed independently by the input/output split.

Representative write surfaces (top-level keys returned by `next_update`):

| Process | Reads (input-only) | Writes (needs output) | Note |
|---|---|---|---|
| `environment/exchange` | `external` | `exchanges` (accumulate), `internal` (accumulate, **no `_updater` declared**) | `internal` is written but declares no updater → don't gate outputs on `_updater` |
| `environment/local_field` | `dimensions`, `location` | `exchanges`, `fields` (both via inline `{_value,_updater}` — see B1) | |
| `antibiotics/conc_to_counts` | `conc`, `volume` (read-only, `set`/`split` dividers) | `bulk` (sparse list) | `conc`/`volume` carry dividers but are read-only → declaring them `overwrite`/typed on input is exactly the A1 hazard |
| `metabolism_redux_classic` | `environment.media_id` (read), `boundary.external.*`, bulk_total, many | `bulk`, `listeners.*`, `environment.exchange`, `next_update_time` | `media_id` is `set`-updater but read-only → the canonical bug; pass `output_ports` excluding it |

Recommended convention: keep `wrap_vivarium_process(output_ports=<written top-level ports>)` per process (mirroring vEcoli's `_output_ports`), **and** make `inputs()` strip `Overwrite`. The first prevents over-declaring outputs; the second prevents the `media_id` input-side structure injection even when a port is legitimately bidirectional.

---

## Verification

`tests/test_translate_ports_audit.py` — 11 passing cases (run with the harness `.venv`). Regression guards for A3/A6/A8 and the empty-collection fallback; explicit "documents current wrong behavior" guards for C1 (`test_divider_is_ignored`), A4 (`test_glob_with_named_sibling_breaks`), and A1 (`test_set_updater_overwrite_applies_to_reads_too`). When a gap is fixed, the corresponding gap-test should be inverted.
