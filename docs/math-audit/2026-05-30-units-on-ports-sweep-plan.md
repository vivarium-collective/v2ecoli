# Units-on-Ports Sweep — Plan & Work-List

**Date:** 2026-05-30
**Branch:** `feat/units-on-ports`
**Goal:** Put the framework's real `quantity[...]` type (a runtime `pint.Quantity`,
not `float[unit]` metadata) on dimensioned ports so the model's shape is explicit
and units are tracked through stores, computation, and serialization.

## Foundation (done)

- **Emitter** (`pbg-emitters#2`): unit-bearing (`Quantity`) ports are supported on emit.
  `strip_quantities` replaces each Quantity leaf with its `.magnitude` **under the same
  column name** (plain-float column — downstream name-based queries/reports keep working)
  and records `column → unit` as file-level Parquet metadata (`column_units` JSON). This
  *supersedes* the first cut (which split into `…__magnitude`/`…__units__` columns and
  thereby renamed the column). Without this, a Quantity on any emitted port crashes the
  ParquetEmitter (`pl.Series` object dtype). **Prerequisite for converting emitted ports.**
  - **Still open for widely-read fields:** reports that read the *live composite state*
    and call `float(value)` (e.g. `reports/multigeneration_report.py`,
    `benchmark_report.py` on `cell_mass`/`dry_mass`) break on a real Quantity —
    `float()` on a dimensioned Quantity raises. The emitter fix only covers *emitted*
    data; live-state `float()` consumers must be migrated to `.to(unit).magnitude` before
    a widely-read field is converted. See T1.
- **Pilot wire** (committed): `effective_elongation_rate`
  (`polypeptide_elongation` → `polypeptide_initiation`), now `quantity[float,amino_acid/s]`
  on both ends. Verified: composite builds + runs through the emitter; round-trip green.

## How a conversion works (the per-field recipe)

1. Type the port `quantity[<mag>,<unit>]` (e.g. `quantity[float,amino_acid/s]`,
   `array`/`overwrite`/`map` wrappers as before).
2. **Writer** emits a real `pint.Quantity` (wrap the computed magnitude:
   `value * units.<u>`), instead of a bare float.
3. **Readers** strip with `.to(<canonical>).magnitude` where they need a float for
   numpy/legacy math (or keep Quantity arithmetic if the consumer is unit-aware).
4. Fix any in-file test stubs to pass Quantities.
5. Gate: build baseline composite + run steps + save-state round-trip.

**Behavior parity:** keep the *canonical* unit (the one the magnitude is already in),
so the stored magnitude is unchanged → outputs identical. Only an actual unit
*conversion* (e.g. mM↔µM) would shift a magnitude and require a behavior re-baseline.
Almost all conversions here are parity-preserving.

## Scope decisions

- **`timestep` (`integer[s]`, ~18 sites): LEAVE as metadata.** It's a control port
  consumed as a bare int in `rate*dt` / `x/dt` arithmetic in nearly every process.
  Promoting it to a Quantity forces `.magnitude` at every use site across the whole
  model — large churn for a unit that is already obvious. Not worth it.
- **Functional cross-process wires (dimensioned values read + used in another process's
  math): PROMOTE.** Highest enforcement value — pint catches a dimension error in the
  consuming computation. (The pilot wire is the first of these.)
- **Config dimensioned params: PROMOTE where used in unit arithmetic.** Most already are
  `quantity[...]` (cellDensity, n_avogadro, elongation_max, Kms, rates, …). Promote the
  stragglers still typed bare `float`/`array[float]` that are genuinely dimensioned.
- **Write-only emitted listeners (concentrations, masses, rates that are *outputs* only):
  PROMOTE (recommended) — gives self-describing emitted columns (unit in Parquet).**
  Lower runtime-enforcement value (not consumed in arithmetic), and costs writer wrapping,
  so these are lower-priority tranches.
- **Dimensionless stays dimensionless:** bulk molecule counts, probabilities,
  stoichiometry matrices, masks, indices.

## Tranches (work-list)

Ordered by value × containment. Each is gated independently.

**T0 — pilot (done):** `effective_elongation_rate`.

**T1 — mass subsystem (`cell_mass` + siblings) — its own project, NOT a tranche.**
Bigger than first scoped: the mass listener computes ~18 interlocking `float[fg]` fields
(`dry_mass = cell_mass − water_mass`, `volume = cell_mass/density`, submasses), so
converting `cell_mass` drags in the whole subsystem. Readers (~10 processes) do
`mass * units.fg` — converting the port means *dropping* that reconstruction (parity-
preserving). BUT it also touches **~111 report/viz references** to `cell_mass`/`dry_mass`:
- *emitted-data* consumers: fixed by the emitter foundation (column name preserved). ✓
- *live-state* consumers doing `float(value)`: **still break** — must migrate to
  `.to(units.fg).magnitude` first.
Plus `division.py` (bare `0.0` mass defaults → `0.0*units.fg`) and the PDMP step
(`float(mass_in...)`). Treat as a dedicated project: migrate live-state `float()`
consumers → convert the mass subsystem → gate (build all 3 architectures + behavior +
the report scripts).

**T2 — config straggler params.** Bare-typed dimensioned config fields not yet
`quantity[...]`. Read-once in `initialize`; contained per process. (Inventory per process
during execution.)

**T3 — metabolism concentration listeners (`mM`):** `conc_updates`,
`target_concentrations`, `counts_to_molar`, `target_aa_conc`. Write-only; wrap the
writer emits in `metabolism.py`.

**T4 — polypeptide_elongation charging concentration listeners (`µM`/`mM`):**
`synthetase_conc`, `*_trna_conc`, `aa_conc`, `ribosome_conc`, `ppgpp_conc`, `rela_conc`,
`spot_conc`, `aa_supply_aa_conc`. Write-only; values computed in the elongation models —
wrap at the listener-write boundary. (Also drop the shadowed duplicate inputs/outputs
block here — same fields declared twice.)

**T5 — remaining `float[nt]` / `float[1/s]` / `float[g/L]` stragglers** across rna_*,
chromosome_*, equilibrium, tcs. Per-field, gated.

## Cross-process wire map (so both ends convert together)

| value | writer | reader(s) | tranche |
|---|---|---|---|
| `effective_elongation_rate` | polypeptide_elongation | polypeptide_initiation | T0 ✓ |
| `cell_mass` | mass listener | ~all (volume calc) | T1 |
| concentration listeners | metabolism / polypeptide_elongation | (emitted only) | T3/T4 |

## T1 execution detail (discovered scope)

Converting `cell_mass` is a **full mass-listener Quantity refactor**, not a few edits —
the fields interlock:

- **`cellDensity`** is a bare `float` (1100.0) config → must become `quantity[g/L]` (or be
  wrapped tolerantly in `initialize`), since `volume = cell_mass / cellDensity`.
- **`volume`** (`float[fL]`) → `quantity[float,fL]` (`(cell_mass/cellDensity).to(units.fL)`).
- **`old_dry_mass`** is read back from the `dry_mass` store each tick → becomes a Quantity,
  so `growth = dry_mass − old_dry_mass` must be Quantity arithmetic.
- **`growth`** (`float[fg/s]` — note: value is a per-tick Δmass in fg, a pre-existing
  mislabel) and **`instantaneous_growth_rate`** (`1/s`) → Quantities.
- **`*_fold_change` / `*_mass_fraction`** (dimensionless `float` ports) → the ratios become
  dimensionless Quantities; `.magnitude`-strip on assignment.
- **stored `*Initial` attrs** (`dryMassInitial`, …) → Quantities (used in the ratios).
- **~10 process readers** do `states[...]["cell_mass"] * units.fg` → drop the `* units.fg`
  (value is already a Quantity).
- **live-state consumers** (`reports/multigeneration_report.py`, `benchmark_report.py`,
  `steps/division.py` defaults, `steps/millard_pdmp_metabolism.py` `float(...)`) → use a
  tolerant `fg_magnitude(x)` helper (accepts float **or** Quantity[fg]).

Execution order (each gated): (1) add tolerant helpers `as_fg_quantity(x)` / `fg_magnitude(x)`
to `v2ecoli/library`; (2) migrate live-state consumers to `fg_magnitude`; (3) convert the
mass listener to Quantity arithmetic (strip ratios with `.magnitude`); (4) convert process
readers; (5) gate: build baseline+departitioned+reconciled, run, behavior suite, + run the
report scripts.

## Runtime mass-conservation check (design — chosen: balance-vs-exchange, emit+warn)

New observe-only Step `MassConservationListener` (wired after metabolism + mass listener):

- **Law:** over one tick, total cell dry-mass change equals net mass imported across the
  environment boundary by metabolism. Everything else (transcription/translation/etc.) only
  repackages existing atoms → internally mass-conserving.
- **Inputs:** `listeners.mass.dry_mass` (Quantity[fg] after T1), the metabolism exchange
  (`environment.exchange` counts, or `fba_results.external_exchange_fluxes` × `coefficient`),
  and exchange-molecule **molecular weights** (from sim_data bulk masses).
- **Residual:** `residual = Δdry_mass − exchange_mass_in` (both in fg). Emit
  `listeners.mass.conservation_residual_fg` and a relative `…_rel` each tick.
- **Reaction:** `warnings.warn(...)` when `|relative residual| > tol` (configurable, e.g.
  1e-3). **Never raises** — observable only.
- **Unit safety:** with T1 done, the residual is Quantity arithmetic (fg vs MW×counts),
  so pint catches a wrong-unit MW or flux at the point of subtraction — the enforcement
  the check is meant to provide.
- **Validation:** on a healthy baseline run the relative residual should sit near machine/
  rounding noise; a large residual flags a leaking process (ties back to the math-audit
  conservation findings).

These two are the next focused builds; both gate on the emitter PR (#2) for the durable env.

## Notes

- Durable env: the venv currently resolves `pbg_emitters` to the source repo via a hand-added
  `.pth`; once `pbg-emitters#2` merges, bump the pin in `uv.lock`/`pyproject` and drop the `.pth`.
- Each tranche is one commit on `feat/units-on-ports`; behavior suite run once at the end
  (and after T1, the only tranche with broad reader changes).
