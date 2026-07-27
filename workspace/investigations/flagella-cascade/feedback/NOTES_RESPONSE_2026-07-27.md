# Response to Maya's "Notes on v2ecoli" (2026-07-27)

Point-by-point response to the four items in `Notes on v2ecoli.docx`.

## 1. Config & analysis files — how do we plot / run the model? Do we have config options?

Yes. v2ecoli runs are config-driven, two levels:

- **Whole pipelines** — `v2ecoli-workflow --config <config.json>` drives multiseed ×
  multigeneration × multivariant sweeps and emits partitioned Parquet + a
  `summary.json` under `out/workflow/`. Example configs live in
  `v2ecoli/configs/` (e.g. `two_generations.json`).
- **Single-cell report** — `.venv/bin/python reports/workflow_report.py` runs one
  cell to division and writes a self-contained `out/workflow/workflow_report.html`.
- **This investigation's figures** — `run_studies.py` /
  `run_studies_multigen.py` / `run_studies_ensemble.py` under
  `workspace/investigations/flagella-cascade/` (feature OFF → study 01; feature ON →
  02/03), each with `--seconds`, `--sample`, `--seed`, `--cache-dir` flags.

Your vEcoli `configs/flagella_*.json` (complexation_flagella, actual_rna_synth_prob,
bulk_monomers, heatmaps, …) are the vEcoli equivalent — in v2ecoli that role is
filled by the run scripts above plus each study's `study.yaml` (`baseline:` block =
composite + params + `features`).

## 2. Initial-state overrides — is this an option in v2ecoli?

Yes — and we already use YOUR exact overrides. Study 03
(`run_low_flagella_gate.py`) applies your `flagellum_initial_value.json` verbatim:

```
CPLX0-7452[j]: 4      # complete flagella
FLAGELLAR-MOTOR-COMPLEX[j]: 0
EG11355-MONOMER[c]: 500   # free FliA
G369-MONOMER[c]: 800      # FlgM
```

That drives the `02_lowIC_gated_cascade` figure. Initial-state overrides are a
first-class v2ecoli feature (set bulk/unique counts on the composite state before
`run()`); the low-IC driver is a worked example you can copy for other conditions.

## 3. workflow_report.py crash (`ParquetEmitter has no attribute 'history'`)

Fixed — **v2ecoli PR #415** (framework change to `main`). Your diagnosis was exactly
right: the report assumed an in-memory `.history` no emitter provides (ParquetEmitter
streams to disk). While fixing it, two more stacked report-only bugs surfaced and are
also fixed:

- `float(dry_mass)` raised a pint `DimensionalityError` (mass listeners are now
  unit-bearing Quantities under units-on-ports) — added a unit-stripping helper.
- the report HTML write raised `UnicodeEncodeError` (▸) under an ASCII locale —
  all report text writes now pin `encoding='utf-8'`.

The report now runs a full cell cycle + both daughters and writes
`out/workflow/workflow_report.html`. (All three were reporting-only; the model /
biology were never affected — as you noted.)

## 4. Did the reconstruction / `.tsv` changes carry over into v2ecoli?

Audited (diffed your `vivarium-collective/vEcoli@biofilm` against its base). Result:

- **There are NO `.tsv` reconstruction changes on `biofilm`** — nothing was edited in
  the flat KB, so there was nothing of that kind to miss.
- The **only** reconstruction change on `biofilm` is a 7-line addition to
  `reconstruction/ecoli/dataclasses/state/internal_state.py` — the
  `init_prob_override: f8` promoter field the SUM-gate writes.
- That change **DID carry over**: v2ecoli's
  `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/state/internal_state.py`
  has `"init_prob_override": "f8"` (verified), so a rebuilt cache carries the column.

So your worry ("probably not, since it's not on vEcoli main") is resolved — the
flagella model needs no base-KB `.tsv` edits, and the one reconstruction field it
does need is present.

---

Also: the flagella investigation branch (PR #276) is now **updated to the latest
v2ecoli `main`** (a real port — main renamed the `baseline` composite to
`ecoli_baseline` and refactored the schema types; the flagella feature was verified
intact afterward).
