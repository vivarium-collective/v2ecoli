# Refactor: `calculate_trna_charging` — accept numpy magnitudes

**Branch:** `refactor/calculate-trna-charging-numpy`
**Sibling perf PR:** #89 (`perf/in-place-fixes`) — landed the bit-equivalent
incremental wins; this is the bigger structural change that was scoped out
of that hunt as too risky for a single-commit edit.

## Why

`v2ecoli/processes/polypeptide/kinetics.py::calculate_trna_charging`
currently advertises a pint-Quantity signature, then strips every input
to a magnitude on the very first lines of the body:

```python
synthetase_conc   = unum_to_pint(synthetase_conc).to(MICROMOLAR_UNITS).magnitude
uncharged_trna_conc = unum_to_pint(uncharged_trna_conc).to(MICROMOLAR_UNITS).magnitude
charged_trna_conc = unum_to_pint(charged_trna_conc).to(MICROMOLAR_UNITS).magnitude
aa_conc          = unum_to_pint(aa_conc).to(MICROMOLAR_UNITS).magnitude
ribosome_conc    = unum_to_pint(ribosome_conc).to(MICROMOLAR_UNITS).magnitude
```

The caller in `polypeptide/elongation_models.py:request()` constructs
those five pint Quantities only because the function signature asks for
them:

```python
synthetase_conc = self.counts_to_molar * synthetase_counts   # → pint Quantity
aa_conc         = self.counts_to_molar * aa_counts
uncharged_trna_conc = self.counts_to_molar * uncharged_trna_counts
charged_trna_conc   = self.counts_to_molar * charged_trna_counts
ribosome_conc       = self.counts_to_molar * ribosome_counts
```

End-to-end: 5 pint Quantities allocated → 5 unit conversions → 5
magnitude extractions. The math is dimensionally trivial — every input is
already in the natural μM basis once `counts_to_molar` is in μM. The
ping-pong is pure overhead.

`ppgpp_metabolite_changes` has the same pattern (7 conversion lines).

## What this refactor does

1. **Change `calculate_trna_charging`** to accept numpy magnitude arrays
   in `MICROMOLAR_UNITS` directly. The signature becomes:

   ```python
   def calculate_trna_charging(
       synthetase_conc_uM:   npt.NDArray[np.float64],
       uncharged_trna_conc_uM: npt.NDArray[np.float64],
       charged_trna_conc_uM:   npt.NDArray[np.float64],
       aa_conc_uM:          npt.NDArray[np.float64],
       ribosome_conc_uM:    npt.NDArray[np.float64],
       f: npt.NDArray[np.float64],
       params: dict[str, Any],
       supply: Optional[Callable] = None,
       time_limit: float = 1000,
       limit_v_rib: bool = False,
       use_disabled_aas: bool = False,
   ) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
   ```

   Drop the 5-line magnitude-extraction prologue.

2. **Update `elongation_models.py:request()`** to compute the conversion
   factor once:

   ```python
   counts_to_uM_mag = self.counts_to_molar.to(MICROMOLAR_UNITS).magnitude
   synthetase_conc_uM   = counts_to_uM_mag * synthetase_counts
   aa_conc_uM           = counts_to_uM_mag * aa_counts
   uncharged_trna_conc_uM = counts_to_uM_mag * uncharged_trna_counts
   charged_trna_conc_uM   = counts_to_uM_mag * charged_trna_counts
   ribosome_conc_uM       = counts_to_uM_mag * ribosome_counts
   ```

   Fix the listener output block at lines 462–491 — currently calls
   `.to(MICROMOLAR_UNITS).magnitude` on values that are now plain μM
   arrays. The values are already in μM, so just pass them through (and
   convert μM → mM = `× 1e-3` for `aa_supply_aa_conc`).

3. **Do the same for `ppgpp_metabolite_changes`** — it takes seven
   pint inputs (uncharged_trna_conc, charged_trna_conc, ribosome_conc,
   rela_conc, spot_conc, ppgpp_conc, counts_to_molar) and strips them
   identically. Update the two call sites inside `elongation_models.py`
   (lines 432 and 630-area) to pass μM magnitudes.

4. **`get_charging_supply_function`** already strips `counts_to_molar`
   internally on its first line; same change there.

5. **Other caller**: `v2ecoli/library/initial_conditions.py` line 2169
   imports a *different* `calculate_trna_charging` (from
   `ecoli.processes.polypeptide_elongation`, the upstream Unum-native
   one) and wraps inputs with `pint_to_unum`. That call is at init
   time (~once), not per-tick. Leave it alone.

## Numerical risk

**Low to moderate.** The arithmetic inside the function is unchanged;
only the front-door stripping moves out. Bit-equivalence at sample
points should hold modulo FP order — and the multiplication
`counts_to_molar.to(MICROMOLAR_UNITS).magnitude × counts_array` performs
the **same** float ops as `(counts_to_molar × counts_array).to(MICROMOLAR_UNITS).magnitude`
modulo associativity. pint internally does the equivalent of the former
when stripping at the boundary, so the result should be identical.

The one thing to watch: the `supply_function` closure produced by
`get_charging_supply_function` uses `counts_to_molar` and `aa_conc` in
the body. Make sure the closure's internal arithmetic is consistent
with the new caller-supplies-magnitudes contract.

## Acceptance

- All `v2ecoli/processes/polypeptide/kinetics.py:calculate_trna_charging`
  + `ppgpp_metabolite_changes` + `get_charging_supply_function` callers
  pass numpy magnitudes; no per-call `unum_to_pint(...).to(...).magnitude`
  inside those functions.
- `scripts/bench_equiv.py` (from PR #89, or copied here) reports
  `equivalence OK at tol_rel=0.005` between this branch and `main`,
  with `dry_mass`, `cell_mass`, `effective_elongation_rate`,
  `fba_objective` matching at every 120 s sample point of a 600 s sim.
- `pytest -m sim tests/test_model_behavior.py` passes.
- Wall-time delta should be ≥ noise floor on the kFBA `calculate_request`
  path. Honest paired-interleaved measurement vs main expected: 1-3%
  (the ~150k `pint.Quantity.__new__` calls in the profile correspond to
  ~0.8 s out of 50 s wall = ~1.5% theoretical ceiling on this lever).

## Why this isn't on PR #89

Tried a *surgical* version on PR #89: added `_to_micromolar_magnitude`
helper with an `isinstance(x, np.ndarray)` fast-path so both pint and
numpy inputs would work, and updated `elongation_models.py:request()` to
pass magnitudes. **Mean wall time was within noise** — the isinstance
dispatch + caller-side `.to(MICROMOLAR_UNITS)` roughly equaled what the
inner conversions saved, and the listener output block (lines 462–491)
needed its own untangling to handle the new magnitude-typed inputs.
Reverted there because the win didn't materialize without the full,
non-isinstance-dispatched refactor in this branch.

## How to start

1. Copy `scripts/bench_baseline.py` + `scripts/bench_equiv.py` from
   PR #89 (or the local `/tmp/bench_equiv.py` snapshot).
2. Record a baseline fingerprint on `main`:
   `.venv/bin/python scripts/bench_equiv.py --out before.json --duration 600`
3. Make the function changes in `kinetics.py` (signature + drop the
   stripping prologue).
4. Update `elongation_models.py:request()` callers; reconcile the
   listener block.
5. `.venv/bin/python scripts/bench_equiv.py --out after.json --duration 600 --baseline before.json --tol-rel 0.005`
6. Iterate until `equivalence OK at tol_rel=0.005` + paired-interleaved
   speedup > noise floor.

## Files touched

```
v2ecoli/processes/polypeptide/kinetics.py
v2ecoli/processes/polypeptide/elongation_models.py
```

Optionally also `scripts/bench_baseline.py` + `scripts/bench_equiv.py`
if PR #89 hasn't merged yet.
