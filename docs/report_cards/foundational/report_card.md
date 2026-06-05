# Foundational behavioral checks — report card

- **Model**: e6f8ea7
- **Stimulus**: single cell from a pre-division checkpoint (no ensemble) · 7 invariant checks via pytest -m behavior
- **Reference status**: populated
- **Generated**: 2026-06-05 13:39

## Overall: PASS (partial) (3 ✓ · 0 ≈ · 0 ✗ · 4 –)

### Growth

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Dry mass roughly doubles | — | assertion holds | — | – ungraded |
| Growth is monotone (no stalls) | — | assertion holds | — | – ungraded |

### Replication

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Replication completes in window | — | assertion holds | — | – ungraded |
| Forks persist mid-cycle | — | assertion holds | — | – ungraded |

### Division

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Division conserves bulk |  | assertion holds | pass | ✓ within_tol |
| Division conserves chromosomes |  | assertion holds | pass | ✓ within_tol |

### Daughter viability

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Daughters build and grow |  | assertion holds | pass | ✓ within_tol |

## Findings

- Foundational invariants, graded as pytest-as-evidence: the result of the `behavior`-marked single-cell checks flows through the shared boolean criterion. A skipped check (missing checkpoint/trajectory) is `ungraded`, not a pass or fail.
- These are preconditions the population (ensemble) cards assume; they compose with those cards (e.g. as a fast pre-merge gate) but are graded as a distinct, single-cell kind of behavioral check.

_Behavioral report card — see docs/report_cards/README.md for the index and how the cards compose._
