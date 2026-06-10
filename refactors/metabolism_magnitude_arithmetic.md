# Refactor: `metabolism.py` — magnitude-arithmetic for per-tick values

**Branch:** `refactor/metabolism-magnitude-arithmetic`
**Sibling refactor PR:** #91 (calculate_trna_charging numpy magnitudes)
**Sibling perf PR:** #89 (in-place fixes, landed bit-equivalent micro-wins)

## Why

`Metabolism._do_update` builds a chain of pint Quantities every tick
from raw float state values, only to strip them all back to magnitudes
at multiple `.to(SOME_UNIT).magnitude` extraction points. Sample
sequence from `v2ecoli/processes/metabolism.py` (line numbers approximate):

```python
# read float state, attach units
cell_mass  = states["listeners"]["mass"]["cell_mass"] * self._fg_unit   # → Quantity[fg]
dry_mass   = states["listeners"]["mass"]["dry_mass"]  * self._fg_unit   # → Quantity[fg]

# Quantity arithmetic
cellVolume       = cell_mass / self.cellDensity                          # → Quantity[L]
counts_to_molar  = (1 / (self.nAvogadro * cellVolume)).to(CONC_UNITS)    # → Quantity[mM]
coefficient      = dry_mass / cell_mass * self.cellDensity * timestep * units.s
                                                                          # → Quantity[g·s/L]

# downstream extracts magnitudes for the actual computations
flux       = (self.ngam * coefficient).to(CONC_UNITS).magnitude          # mM
flux       = (counts_to_molar * gtp_to_hydrolyze).to(CONC_UNITS).magnitude
fluxes_out = (exchange_fluxes / coefficient).to(GDCW_BASIS).magnitude

# and listener serialisation
coefficient_magnitude_for_listener = coefficient.to(CONVERSION_UNITS).magnitude
```

Most of these `.to(X).magnitude` operations are **constant-factor
multiplications** dressed up as dimensional analysis: the conversion
ratio between two unit bases is a compile-time constant once you fix
the source and target units.

The hot-path cost is real but not dominant — `pint.facets.plain.quantity.py:186(__new__)`
accounts for ~0.8 s of a 50 s sim. The bigger payoff is **clarity**:
the actual numerics are pure float arithmetic; the unit machinery is
documentation that runs at runtime.

## Numerical risk

**Moderate to high.** Two failure modes that are easy to make:

1. **Forgetting a unit conversion factor.** If `coefficient` is computed
   as `dry_mass_fg * timestep_s * cellDensity_g_per_L / cell_mass_fg`,
   the result is in `g·s/L`. To get `mmol/g/h` (GDCW_BASIS) you need
   `× 1/3600` for s→h. Missing one of these silently scales every
   FBA flux by a constant factor — growth rate breaks in ways that
   may not be caught by `bench_equiv.py` at first.

2. **Boundary with Unum-native upstream code.** `exchange_constraints`,
   `get_import_constraints`, and `get_kinetic_constraints` are
   upstream-Unum (Birch et al. legacy). They accept and return Unum
   Quantities. The refactor needs to rebuild Quantities at exactly
   these boundaries — not before, not after.

Both failure modes require careful per-line verification with a
dimensional bookkeeping table.

## What this refactor does

1. **At init**, strip every constant pint Quantity to a magnitude in a
   documented unit basis:

   ```python
   self._cell_density_g_per_L = float(self.cellDensity.to(units.g / units.L).magnitude)  # 1100.0
   self._n_avogadro_per_mol   = float(self.nAvogadro.to(1 / units.mol).magnitude)        # 6.022e23
   self._ngam_mmol_per_g_per_h = float(self.ngam.to(units.mmol / units.g / units.h).magnitude)  # 8.39
   ```

2. **At the top of `_do_update`**, compute per-tick scalars as plain
   floats with documented basis comments:

   ```python
   # State: cell_mass and dry_mass come from the listener as plain floats in fg.
   cell_mass_fg = states["listeners"]["mass"]["cell_mass"]
   dry_mass_fg  = states["listeners"]["mass"]["dry_mass"]
   timestep_s   = states["timestep"]

   # Derived geometry
   cell_volume_L          = cell_mass_fg * 1e-15 / self._cell_density_g_per_L
   counts_to_molar_M      = 1.0 / (self._n_avogadro_per_mol * cell_volume_L)
   counts_to_molar_mM_mag = counts_to_molar_M * 1e3       # CONC_UNITS = mmol/L = mM

   # FBA flux→concentration conversion factor (g·s/L)
   coefficient_gsperL = (dry_mass_fg / cell_mass_fg) * self._cell_density_g_per_L * timestep_s
   ```

3. **Replace `.to(X).magnitude` extractions** with explicit factor
   multiplications, with a comment naming each factor:

   ```python
   # GDCW_BASIS = mmol/g/h; coefficient is in g·s/L; factor is s/h = 1/3600.
   exchange_fluxes_gdcw = exchange_fluxes_mM / coefficient_gsperL / 3600.0

   # NGAM has units mmol/g/h; coefficient has g·s/L; product → mM with × 1/3600.
   flux_ngam_mM = self._ngam_mmol_per_g_per_h * coefficient_gsperL / 3600.0
   ```

4. **At the upstream-Unum boundary**, rebuild Quantities exactly once
   per call. Likely needs a small helper:

   ```python
   def _build_coefficient_quantity(coefficient_gsperL):
       """Used only at the boundary with exchange_constraints /
       get_import_constraints / get_kinetic_constraints — they are
       Unum-native and need dimensioned input."""
       return (coefficient_gsperL * units.g * units.s / units.L)
   ```

5. **Listener outputs** that previously serialised `.to(X).magnitude`
   now serialise the magnitude floats directly. Watch the listener
   schema — some fields advertise specific units; the schema is the
   source of truth.

## Acceptance

- All per-tick pint Quantity construction in `Metabolism._do_update`
  hot path is replaced by float arithmetic.
- pint Quantity is built **only** at the upstream-Unum boundary
  (`exchange_constraints`, `get_import_constraints`,
  `get_kinetic_constraints`).
- `scripts/bench_equiv.py` reports `equivalence OK at tol_rel=0.005`
  between this branch and `main` (use a tighter tolerance if practical;
  this is a numerically sensitive area).
- `pytest -m sim tests/test_model_behavior.py` passes — especially:
  - `test_baseline_dry_mass_roughly_doubles`
  - `test_baseline_monotone_growth`
  - `test_replication_completes_in_expected_window`
- A dimensional bookkeeping table is included alongside the code
  (either in this doc or as a comment block in metabolism.py) naming
  every conversion factor and its derivation.
- Paired-interleaved 600 s sim wall-time speedup > noise floor.

## Dimensional bookkeeping (skeleton)

Fill this in as the refactor proceeds — keeping the table in this doc
or in metabolism.py is the deliverable that prevents factor-of-3600 bugs.

| Quantity | Basis (this refactor) | Source basis | Conversion |
|---|---|---|---|
| `cell_mass_fg`, `dry_mass_fg` | float, fg | state | 1.0 |
| `cell_volume_L` | float, L | derived | `cell_mass_fg × 1e-15 / cell_density_g_per_L` |
| `n_avogadro_per_mol` | float, 1/mol | const | 6.022e23 |
| `counts_to_molar_M` | float, M (mol/L) | derived | `1 / (n_avogadro × cell_volume_L)` |
| `counts_to_molar_mM_mag` | float, mM | CONC_UNITS | `counts_to_molar_M × 1e3` |
| `coefficient_gsperL` | float, g·s/L | derived | `(dry/cell) × density × timestep_s` |
| `ngam_mmol_per_g_per_h` | float, mmol/g/h | const | 8.39 |
| `exchange_fluxes_gdcw` | float, GDCW_BASIS=mmol/g/h | derived | `flux_mM / coefficient_gsperL / 3600` |
| `flux_ngam_mM` | float, mM | derived | `ngam × coefficient_gsperL / 3600` |

## Boundary functions (rebuild Quantities here, nowhere else)

These three upstream functions are Unum-native and require dimensioned
inputs:

- `self.exchange_constraints(...)` (line ~890)
- `self.get_import_constraints(...)` (line ~530)
- `self.get_kinetic_constraints(...)` (line ~1020)

The refactor passes pint Quantities (via `pint_to_unum`) only at these
exact call sites. Everywhere else is magnitude arithmetic.

## How to start

1. Copy `scripts/bench_baseline.py` + `scripts/bench_equiv.py` from
   PR #89, or wait for that PR to merge.
2. Record baseline fingerprint on `main` (use tightest tolerance you
   can — try `--tol-rel 0.001` first; relax only if FP order can't be
   preserved):
   `.venv/bin/python scripts/bench_equiv.py --out before.json --duration 600`
3. Refactor `Metabolism.initialize` to add the `_cell_density_g_per_L`
   et al. magnitude attributes.
4. Refactor `Metabolism._do_update` top section to compute per-tick
   floats as documented above.
5. Walk through every `.to(SOME_UNIT).magnitude` extraction; replace
   with the documented constant-factor multiply.
6. At each upstream-Unum boundary, rebuild Quantities via
   `pint_to_unum(magnitude × pint_unit)`.
7. Run `pytest -m sim tests/test_model_behavior.py` after every
   logical chunk of changes.
8. Run `bench_equiv.py` at the end at `tol_rel=0.005` or tighter.

## Files touched

```
v2ecoli/processes/metabolism.py     (primary)
docs/refactors/metabolism_magnitude_arithmetic.md  (bookkeeping)
```

## Why this isn't on PR #89

This is the highest-numerical-risk lever I could see. A wrong factor
of 3600 in the FBA flux conversion silently scales every metabolic flux
— growth could survive the change while the underlying biology is
broken in ways that `bench_equiv.py` at `tol_rel=0.005` over a 600 s
run might not catch. The PR #89 hunt was self-described as "find one
bit-equivalent line to change"; this refactor wants a deliberate
day-of-work with per-line dimensional verification.
