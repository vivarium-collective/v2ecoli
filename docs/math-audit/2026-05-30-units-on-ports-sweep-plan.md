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

## Notes

- Durable env: the venv currently resolves `pbg_emitters` to the source repo via a hand-added
  `.pth`; once `pbg-emitters#2` merges, bump the pin in `uv.lock`/`pyproject` and drop the `.pth`.
- Each tranche is one commit on `feat/units-on-ports`; behavior suite run once at the end
  (and after T1, the only tranche with broad reader changes).
