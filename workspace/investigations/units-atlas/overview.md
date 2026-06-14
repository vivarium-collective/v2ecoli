# Units Atlas

A small, descriptive investigation cataloging every unit-bearing readout
across the E. coli whole-cell simulation: which observables are measured, in
what units, grouped by physical dimension (mass, time, concentration, rate,
count, volume, length), with example magnitudes and ranges sampled from a real
baseline run.

Source of truth: the declared `quantity[...]` / `float[unit]` port types,
resolved live via `v2ecoli.library.units_resolver.build_units_index`. No
acceptance gates — this is a reference, not a hypothesis test.
