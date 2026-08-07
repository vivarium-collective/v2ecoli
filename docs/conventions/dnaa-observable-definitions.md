# Canonical DnaA observable definitions

**Status:** convention (enforced by `scripts/check_dnaa_fraction_sources.py`).
**Origin:** Rashmi flagged (2026-06-20) that the dnaa-3 readouts chart showed the
DnaA-ATP ratio "off" (swinging 0→1) while dnaa-4 showed it correctly in band — the
*same run*, but two render scripts computed the fraction two different ways.

## The rule

There is exactly **one** definition of the DnaA-ATP / DnaA-ADP fraction, and it
lives in **`scripts/dnaa_observables.py`**. Every render/analysis script that
reports a DnaA nucleotide-state fraction (or the total DnaA pool) MUST import that
module and use `decompose(...)` / `decompose_from_frame(...)`. Do not re-derive it
inline.

```python
import dnaa_observables as dnaa
dec = dnaa.decompose_from_frame(df)   # df has apo/atp/adp + BOUND_*_COLS
atp_fraction = dec["atp_fraction"]    # (free ATP + Σ bound ATP) / total DnaA
```

## The definition (and why it matters)

```
DnaA-ATP fraction = (free DnaA-ATP + Σ bound-ATP) / total DnaA
total DnaA        = apo + free-ATP + free-ADP + Σ bound-ATP + Σ bound-ADP
```

The fraction is over the **whole** pool — free **plus** the DnaA bound at the
chromosomal / oriC / promoter boxes (the `listeners.replication_data.*_bound_atp/adp`
columns). It is **NOT** the free-only fraction `free-ATP / (free pools)`.

**The trap:** before dnaa-3 there is no box binding, so the bulk pools *are* the
total and free-only happens to be correct. From dnaa-3 onward the **active
Langmuir binding depletes the free pool** — on the dnaa-4 reference run DnaA-ATP is
~98 % bound (≈94 bound vs ≈2 free). The free-only fraction then divides by a
near-empty denominator and swings 0→1 (or `nan`), which is exactly the contradictory
chart Rashmi caught. The total (free+bound) fraction sits stably in the Boesen
[0.2, 0.5] band.

## Presentation

The per-tick total fraction still oscillates within the cycle (real charging
dynamics). When showing it against the [0.2, 0.5] band, overlay the
**per-generation mean** so the in-band signal is unambiguous (see
`render_dnaa3_readouts.py` panel 2 and `render_dnaa2_atp_band.py`).

## Enforcement

`scripts/check_dnaa_fraction_sources.py` scans `scripts/render_dnaa*.py` and fails
if a script computes a DnaA nucleotide fraction without importing
`dnaa_observables`. Run it in the report-prep flow. Scripts that legitimately
predate box binding (dnaa-0/1/2) or plot box *occupancy* (not the nucleotide
fraction) are listed in the checker's allowlist with a reason.
