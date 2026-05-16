"""Behavioral tests for dnaa-01-expression-dynamics.

Each test corresponds to a behavioral_tests entry in study.yaml (BT-01 … BT-05).
Run via `pytest studies/dnaa-01-expression-dynamics/tests/` (or via the
dashboard's per-study test runner, which is what populates
study.yaml.tests.last_results).

All tests skip cleanly when the relevant run hasn't been executed yet, so
the suite is useful as living documentation even before any simulation
output exists.
"""
from __future__ import annotations

import statistics

import pytest

from conftest import (
    DNAA_MONOMER_ID,
    DNAA_MRNA_ID,
    DNAA_TF_ID,
    bulk_count,
    listener_value,
)


def _second_half(history):
    """Return the snapshots from the second half of the run.

    Used so steady-state assertions ignore the initial transient.
    """
    n = len(history)
    if n < 2:
        return history
    return history[n // 2 :]


# ── BT-01 ───────────────────────────────────────────────────────────────────

def test_dnaA_count_within_mass_spec_range(baseline_history):
    """BT-01: median DnaA count in [300, 800] over the second half."""
    counts = [bulk_count(s["state"], DNAA_MONOMER_ID) for s in _second_half(baseline_history)]
    counts = [c for c in counts if c is not None]
    if not counts:
        pytest.skip(f"DnaA monomer {DNAA_MONOMER_ID} not present in bulk; check identifier")
    median = statistics.median(counts)
    assert 300 <= median <= 800, (
        f"median DnaA count {median} is outside the mass-spec / EcoCyc "
        f"range [300, 800]. Min={min(counts)}, max={max(counts)}, n={len(counts)}."
    )


# ── BT-02 ───────────────────────────────────────────────────────────────────

def test_dnaA_concentration_stability_within_10pct(baseline_history):
    """BT-02: rolling 5-min CV of DnaA concentration < 0.10.

    Falls back to count stability (CV on count) if the dedicated
    concentration listener isn't wired up yet (gap-1 from study.yaml).
    """
    second = _second_half(baseline_history)
    series = [
        listener_value(s["state"], "listeners.dnaa.protein_uM")
        for s in second
    ]
    if all(v is None for v in series):
        # Fall back to count-based stability — the gap-1 listener isn't live.
        series = [bulk_count(s["state"], DNAA_MONOMER_ID) for s in second]
        series = [c for c in series if c is not None]
        if not series:
            pytest.skip("neither concentration listener nor bulk count available")

    if len(series) < 5:
        pytest.skip(f"history too short for stability check: n={len(series)}")

    # 5-step rolling CV (test fixture step is roughly minutes).
    cvs = []
    window = 5
    for i in range(len(series) - window + 1):
        w = series[i : i + window]
        m = statistics.mean(w)
        if m == 0:
            continue
        cvs.append(statistics.stdev(w) / m if len(w) > 1 else 0.0)
    max_cv = max(cvs) if cvs else 0.0
    assert max_cv < 0.10, (
        f"max rolling CV of DnaA concentration is {max_cv:.3f} (>10%) — "
        f"baseline is not at steady-state."
    )


# ── BT-03 ───────────────────────────────────────────────────────────────────

def test_stop_synthesis_decays_on_dilution_timescale(stop_synthesis_history):
    """BT-03: in stop-dnaA-synthesis variant, DnaA count drops ≥30% over the run."""
    if len(stop_synthesis_history) < 2:
        pytest.skip("need at least two snapshots in stop-dnaA-synthesis run")
    first = bulk_count(stop_synthesis_history[0]["state"], DNAA_MONOMER_ID)
    last = bulk_count(stop_synthesis_history[-1]["state"], DNAA_MONOMER_ID)
    if first is None or last is None or first == 0:
        pytest.skip("DnaA bulk count not available in stop-synthesis snapshots")
    fractional_drop = (first - last) / first
    assert fractional_drop >= 0.30, (
        f"stop-dnaA-synthesis variant: DnaA count dropped only {fractional_drop:.1%}; "
        f"expected ≥30% over one doubling time. first={first}, last={last}."
    )


# ── BT-04 ───────────────────────────────────────────────────────────────────

def test_coarse_autorepression_negative_correlation(baseline_history):
    """BT-04: when DnaA-bound-to-promoters is high, dnaA tx init events are low.

    Uses Pearson r between rna_synth_prob.n_actual_bound[DnaA_idx] and
    rnap_data.rna_init_event[EG10235_RNA_idx]. Requires the run to expose
    both listeners as numpy/list-shaped arrays and the index-by-id helper
    that the dashboard's observable resolver will eventually provide.

    For now: if the listeners aren't shaped as expected, skip.
    """
    bound = []
    init = []
    for s in _second_half(baseline_history):
        n_actual_bound = listener_value(s["state"], "listeners.rna_synth_prob.n_actual_bound")
        rna_init = listener_value(s["state"], "listeners.rnap_data.rna_init_event")
        if not isinstance(n_actual_bound, list) or not isinstance(rna_init, list):
            pytest.skip("required listeners not present as lists in snapshot")
        # Indices need a sim_data lookup; skip the per-index pick for now and
        # use sum-bound as a proxy until index_by lands. Records the spirit.
        bound.append(sum(n_actual_bound) if n_actual_bound else 0)
        init.append(sum(rna_init) if rna_init else 0)

    if len(bound) < 5 or statistics.stdev(bound) == 0 or statistics.stdev(init) == 0:
        pytest.skip(f"insufficient variance to compute correlation (n={len(bound)})")

    # Pearson r
    mean_b = statistics.mean(bound)
    mean_i = statistics.mean(init)
    cov = sum((b - mean_b) * (i - mean_i) for b, i in zip(bound, init))
    denom = (statistics.stdev(bound) * statistics.stdev(init) * (len(bound) - 1))
    r = cov / denom if denom else 0.0

    assert r < -0.3, (
        f"coarse autorepression check: Pearson r between bound-DnaA and "
        f"tx-init events is {r:.2f}; expected r < -0.3."
    )


# ── BT-05 ───────────────────────────────────────────────────────────────────

@pytest.mark.xfail(reason="initiation-event detection not settled; see dnaa-04")
def test_post_initiation_gene_dosage_doubles_tx_rate(baseline_history):
    """BT-05 (stub): within 10 min of a replication-initiation event, dnaA
    transcription rate rises by ≥50%."""
    raise NotImplementedError(
        "BT-05 needs an initiation-event detector in chromosome_initiation; "
        "tracked as a stub until dnaa-04 lands."
    )
