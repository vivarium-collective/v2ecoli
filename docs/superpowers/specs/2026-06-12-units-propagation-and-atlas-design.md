# Units propagation into v2ecoli visuals + Units Atlas investigation

**Date:** 2026-06-12
**Status:** Design approved, pending implementation plan

## Goal

Units declared in process/listener port schemas (`quantity[float,fg]`,
`array[float[mM]]`, `float[1/s]`, …) should automatically appear on plot
axes across *all* v2ecoli visualizations and analyses, with **zero per-plot
code**, resolved **live from the typed schema**. Additionally, ship a small
descriptive **Units Atlas** investigation that catalogs every unit value and
readout across the E. coli simulation.

## Background / constraints discovered

- Units live in port schemas as the `Quantity` bigraph-schema type
  (`v2ecoli/types/quantity.py`), declared via `quantity[<mag>,<unit>]`
  syntax. The unit string is held in the `_units` field.
  Listener ports carry units widely: `cell_mass` in `fg`, `reaction_rates`
  in `1/s`, concentrations in `mM`, etc.
- **Units are stripped at emit time** — parquet/xarray/sqlite store bare
  magnitudes only. So at plot time the numeric data carries no unit.
- Axis labels today are **100% hardcoded** across every Visualization and
  Analysis. None derive from port paths, schema, or emitted metadata.
- An emit-time schema walker already exists
  (`v2ecoli/library/output_metadata.py::_extract_labels_recursive`) that
  traverses typed port schemas to harvest element labels — but it ignores
  units. This is the pattern to mirror for unit extraction.
- At **post-hoc render** (`v2ecoli/library/parquet_viz.py::
  render_study_visualizations`) there is **no composite in scope** — only
  the parquet dataframe and each viz's `inputs_map` (port name → observable
  dotted path, e.g. `listeners.mass.cell_mass`). The schema must be sourced
  another way.
- The Visualization/Analysis base class lives in **`pbg_superpowers`**
  (shared across all workspaces); the unit types are **v2ecoli-specific**.

## Design decisions (locked)

1. **Mechanism:** resolve units **live from the typed schema** — no
   persisted units sidecar.
2. **Auto-labeling:** a **base-class hook** so existing plots get units with
   zero per-plot code.
3. **Hook location:** the hook lives in the **shared `pbg_superpowers`
   Visualization/Analysis base**, with a **pluggable `units_resolver`**;
   v2ecoli supplies the resolver. Other workspaces (resolver `None`) are
   unaffected.
4. **Investigation:** a **Units Atlas / readout catalog** — lightweight and
   descriptive, no pass/fail gates.

## Architecture

Three pieces.

### Piece 1 — Unit resolution (v2ecoli)

New module `v2ecoli/library/units_resolver.py`:

- `build_units_index(schema, core) -> dict[str, str]`
  Walk a typed composite schema tree once, emitting `dotted_path -> unit_str`
  for every leaf whose type is a `Quantity` (read its `_units`). Mirrors the
  traversal in `output_metadata._extract_labels_recursive` so labels and
  units derive from the same pattern. Example entries:
  `"listeners.mass.cell_mass" -> "fg"`,
  `"listeners.equilibrium_listener.reaction_rates" -> "1/s"`,
  `"listeners.fba_results.conc_updates" -> "mM"`.

- `resolve_unit(units_index, path) -> str | None`
  Look up the unit for an observable path. Tolerates array-element and
  sub-leaf paths; returns `None` for unitless paths (`global_time`, raw
  counts).

- `format_axis_label(base_label, unit) -> str`
  `"Mass" + "fg" -> "Mass (fg)"`. **Idempotent** — never produces
  `"Mass (fg) (fg)"`. Returns `base_label` unchanged when `unit` is `None`.

- `V2EcoliUnitsResolver`
  Builds the index **once** from the v2ecoli composite's *static declared*
  port schema — introspected from registered process port declarations / the
  `EcoliWCM` interface, **without a sim run or ParCa load**. Answers
  `resolve_unit(path)`. This is the "live from schema" source: it reads the
  actual declared types, not a persisted snapshot.

### Piece 2 — Base-class hook (pbg_superpowers, pluggable)

In `pbg_superpowers` `visualization.py` (and the `Analysis` base):

- Add a class attribute `units_resolver = None` and a method
  `finalize_axes(...)` that runs after a subclass builds its figure. It
  appends units to axis titles using `units_resolver` + the viz's
  `inputs_map` (port → observable path). Handles both backends:
  - **matplotlib** — rewrite `ax.set_xlabel` / `ax.set_ylabel`.
  - **plotly** — rewrite `layout.xaxis.title` / `layout.yaxis.title`.
- v2ecoli registers its resolver onto the base at import:
  `Visualization.units_resolver = V2EcoliUnitsResolver()`. All v2ecoli
  visualizations then get units for free. Elsewhere the resolver stays
  `None` → the hook is a **no-op**.
- A viz overrides `finalize_axes` to **opt out** or to remap a specific axis
  to a specific observable.
- Deterministic, no AI dependencies — consistent with the dashboard AI-free
  principle.

### Piece 3 — Units Atlas investigation (v2ecoli workspace)

`workspace/investigations/units-atlas/` with one descriptive study:

- **Catalog builder** — run `build_units_index` over the full composite
  schema, then sample one real run to attach an example magnitude + min/max
  per readout. Group readouts by physical dimension: mass / time /
  concentration / rate / count / length / dimensionless.
- **`UnitsAtlasVisualization`** — render the grouped table (dimension →
  readouts → unit, example value, range), plus a **flag list** of readouts
  that are dimensionless or missing a unit, so gaps surface.
- Lightweight: descriptive reference only, no acceptance gates.

## Data flow

Declared `quantity[...]` types → `build_units_index` walks the typed schema
→ `units_index: path → unit` → `V2EcoliUnitsResolver.resolve_unit(path)` →
base-class `finalize_axes` appends the unit to each mapped axis → rendered
HTML/PNG shows `Mass (fg)`. The Atlas reuses the same index plus a run
sample.

## Edge cases

- Unitless paths (`global_time`, counts) → label unchanged (no `(None)`).
- Array-valued readouts (per-reaction rates) → unit applies to the whole
  series.
- Idempotent labeling so re-renders don't stack `(fg) (fg)`.
- Derived-quantity axes (fold-change, fraction, ratio) → the base only labels
  axes it can confidently map via `inputs_map`; ambiguous axes are left
  alone.
- Units shown verbatim (`1/s`, `mM`, `g*s/L`) — no pretty-printing or
  conversion in v1.

## Testing

- Unit tests for `build_units_index` (known listener ports → expected
  units), `resolve_unit` (hit / miss / array / sub-path), and
  `format_axis_label` (append / idempotent / None passthrough).
- A base-hook test: stub matplotlib + plotly figures with a stub resolver,
  asserting axis titles get the unit appended and that override opt-out
  works.
- An Atlas test asserting the catalog covers every `quantity[...]` leaf in
  the composite and flags dimensionless / missing-unit readouts.

## Out of scope (YAGNI)

- No persisted units sidecar (rejected in favor of live resolution).
- No dimensional-consistency gates (the Atlas is descriptive only).
- No unit conversion or normalization — declared units are displayed as-is.
- No changes to non-v2ecoli workspaces' visuals.
