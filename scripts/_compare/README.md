# vEcoli ↔ v2ecoli comparison harness

Run both engines from one vEcoli config and produce a two-column HTML report.

## Usage

The vEcoli side runs ParCa (standalone) and a 2-generation lineage (Nextflow).
That imposes two environment requirements that must be exported before running:

- **Java 17+** for Nextflow (the vEcoli `main.nf` needs Nextflow ≥24, which
  dropped Java 11). Install e.g. `brew install openjdk@21`.
- **vEcoli's venv on PATH**, so the bare `python` inside Nextflow tasks
  resolves to the interpreter that has vEcoli's deps (fsspec, polars, …).

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH="/Users/eranagmon/code/vEcoli/.venv/bin:$JAVA_HOME/bin:$PATH"

.venv/bin/python scripts/compare_harness.py \
    --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
    -o out/compare/report.html
```

`--fast-plumbing` runs ParCa in fast mode for wiring iteration only and
stamps the report NOT SCIENTIFICALLY VALID. Omit it for the real comparison
(full ParCa + two 2-gen lineages — tens of minutes; results cached under
`out/compare_harness/` and reused on rerun, so re-renders are seconds).

### Known external dependencies / patches

- vEcoli's `ecoli/library/parquet_emitter.py` passes `storage_options` (int
  retry values) to polars `write_parquet` unconditionally, which crashes
  local writes on polars ≥1.3 (`'int' object cannot be converted to PyString`).
  A guard to only pass it for cloud URIs is required; upstream this in vEcoli.
- Reused on-disk outputs: a full v2 ParCa at `out/sim_data_full/` can be
  pre-seeded into `out/compare_harness/v2_parca/` (symlink the checkpoints +
  write the run token into `.done`) to skip re-running it.

## Sections

1. **Config & schema diff** — how the vEcoli config maps to v2ecoli
   (adapter in `config_adapter.py`; v2ecoli core untouched).
2. **ParCa / sim_data** — per-step diff via `scripts/parca_compare.py`
   plus a final-sim_data field-by-field diff (tight tolerance).
3. **2-generation sim dynamics** — mass/growth, molecule counts, listeners,
   division/lineage, compared with per-metric tolerances + KS.

If a stage fails (e.g. an engine errors or an output is missing), that
section shows the error and the rest of the report still renders.

## Tests

    .venv/bin/python -m pytest tests/compare/

The full cross-engine run is gated: `COMPARE_E2E=1 .venv/bin/python -m pytest tests/compare/test_end_to_end.py`.
