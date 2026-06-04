# vEcoli ↔ v2ecoli comparison harness

Run both engines from one vEcoli config and produce a two-column HTML report.

## Usage

    .venv/bin/python scripts/compare_harness.py \
        --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
        -o out/compare/report.html

`--fast-plumbing` runs ParCa in fast mode for wiring iteration only and
stamps the report NOT SCIENTIFICALLY VALID. Omit it for the real comparison
(full ParCa, hours; results cached under `out/compare_harness/`).

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
