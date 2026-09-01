"""Conservation invariants for the promoter-keyed synthesis probabilities.

Phase 1 of the promoter/transcript split
(docs/promoter_transcript_split_scope.html). ``build_promoter_keyed_probs``
splits each transcript's synthesis probability across the promoters that
drive it. The split must be *exactly* conservative — that is what makes
the eventual flip to promoter-keyed indexing behaviour-preserving in
aggregate.

These run against a synthetic sim_data stand-in so they stay fast and
deterministic; the same invariants are checked against a real ParCa state
in ``test_conservation_on_real_state`` when one is available.
"""

import collections
import glob
import os
import types

import numpy as np
import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.process.transcription_regulation import (
    TranscriptionRegulation,
)


def _fake_sim_data(rna_ids, promoter_records, per_promoter_ratios=()):
    """Minimal stand-in exposing only what the builder reads."""
    transcription = types.SimpleNamespace(
        rna_data={"id": np.array([f"{r}[c]" for r in rna_ids])},
        _per_promoter_ratios=list(per_promoter_ratios),
    )
    getter = types.SimpleNamespace(
        get_promoter_records=lambda: list(promoter_records)
    )
    return types.SimpleNamespace(
        process=types.SimpleNamespace(transcription=transcription),
        getter=getter,
    )


def _build(rna_ids, promoter_records, basal_prob, delta=(), ratios=(), n_tf=2):
    tr = TranscriptionRegulation.__new__(TranscriptionRegulation)
    tr.basal_prob = np.asarray(basal_prob, dtype=float)
    tr.delta_prob = {
        "deltaI": np.asarray([d[0] for d in delta], dtype=np.int64),
        "deltaJ": np.asarray([d[1] for d in delta], dtype=np.int64),
        "deltaV": np.asarray([d[2] for d in delta], dtype=float),
        "shape": (len(rna_ids), n_tf),
    }
    tr.build_promoter_keyed_probs(
        _fake_sim_data(rna_ids, promoter_records, ratios)
    )
    return tr


def _rec(pid, transcript):
    return {
        "id": pid,
        "transcript_id": transcript,
        "coordinate": 100,
        "direction": "+",
        "gene_tuple": ("g1",),
    }


def test_uniform_split_conserves_basal_prob():
    """Three promoters on one transcript, no curated shares -> 1/3 each."""
    tr = _build(
        rna_ids=["T1"],
        promoter_records=[_rec("T1", "T1"), _rec("P2", "T1"), _rec("P3", "T1")],
        basal_prob=[0.9],
    )
    assert len(tr.promoter_basal_prob) == 3
    assert np.allclose(tr.promoter_initiation_share, 1 / 3)
    assert tr.promoter_basal_prob.sum() == pytest.approx(0.9)


def test_curated_shares_are_used_and_normalised():
    ratios = [
        {"TU_id": "T1", "condition": "basal", "ratio": "0.7"},
        {"TU_id": "P2", "condition": "basal", "ratio": "0.3"},
    ]
    tr = _build(
        rna_ids=["T1"],
        promoter_records=[_rec("T1", "T1"), _rec("P2", "T1")],
        basal_prob=[1.0],
        ratios=ratios,
    )
    by = {pid: i for i, pid in enumerate(tr.promoter_ids)}
    assert tr.promoter_initiation_share[by["T1"]] == pytest.approx(0.7)
    assert tr.promoter_initiation_share[by["P2"]] == pytest.approx(0.3)
    assert tr.promoter_basal_prob.sum() == pytest.approx(1.0)


def test_partial_curated_shares_fall_back_to_uniform():
    """A group is only apportioned when *every* member has a curated row."""
    ratios = [{"TU_id": "T1", "condition": "basal", "ratio": "0.9"}]
    tr = _build(
        rna_ids=["T1"],
        promoter_records=[_rec("T1", "T1"), _rec("P2", "T1")],
        basal_prob=[1.0],
        ratios=ratios,
    )
    assert np.allclose(tr.promoter_initiation_share, 0.5)


def test_delta_prob_is_split_by_the_same_share():
    """Per (TU, TF), the promoter-keyed deltas must re-sum to the TU value."""
    tr = _build(
        rna_ids=["T1", "T2"],
        promoter_records=[
            _rec("T1", "T1"), _rec("P2", "T1"), _rec("T2", "T2"),
        ],
        basal_prob=[0.6, 0.4],
        delta=[(0, 0, -0.12), (0, 1, 0.08), (1, 0, 0.05)],
    )
    per_tu = collections.defaultdict(float)
    for i, j, v in zip(
        tr.promoter_delta_prob["deltaI"],
        tr.promoter_delta_prob["deltaJ"],
        tr.promoter_delta_prob["deltaV"],
    ):
        per_tu[(int(tr.promoter_to_TU[i]), int(j))] += v
    assert per_tu[(0, 0)] == pytest.approx(-0.12)
    assert per_tu[(0, 1)] == pytest.approx(0.08)
    assert per_tu[(1, 0)] == pytest.approx(0.05)
    assert tr.promoter_delta_prob["shape"] == (3, 2)


def test_promoter_absent_from_rna_data_is_dropped():
    """A promoter whose transcript is not modelled contributes nothing."""
    tr = _build(
        rna_ids=["T1"],
        promoter_records=[_rec("T1", "T1"), _rec("PX", "TX_not_modelled")],
        basal_prob=[0.5],
    )
    assert tr.promoter_ids == ["T1"]
    assert tr.promoter_basal_prob.sum() == pytest.approx(0.5)


def test_self_mapping_while_the_dedup_exemption_is_live():
    """Transitional: a promoter that is itself a transcript maps to itself.

    While ``per_promoter_ratios.tsv`` exempts a gene tuple from dedup, its
    TUs are separate transcripts in ``rna_data``. Each must then own its
    synthesis probability outright rather than take a share of a sibling's.
    """
    tr = _build(
        rna_ids=["T1", "P2"],
        promoter_records=[_rec("T1", "T1"), _rec("P2", "T1")],
        basal_prob=[0.6, 0.4],
    )
    assert np.allclose(tr.promoter_initiation_share, 1.0)
    assert np.allclose(sorted(tr.promoter_basal_prob), [0.4, 0.6])


_STATES = sorted(glob.glob("out/sim_data_*/parca_state.pkl.gz"))


@pytest.mark.skipif(not _STATES, reason="no ParCa state on disk")
def test_conservation_on_real_state():
    from v2ecoli.processes.parca.data_loader import (
        load_parca_state, hydrate_sim_data_from_state)

    state = None
    for path in reversed(_STATES):
        sd = hydrate_sim_data_from_state(load_parca_state(path))
        if getattr(sd.process.transcription_regulation,
                   "promoter_basal_prob", None) is not None:
            state = sd
            break
    if state is None:
        pytest.skip("no state carries promoter-keyed arrays")

    tr = state.process.transcription_regulation
    basal = np.asarray(tr.basal_prob)
    promoter_basal = np.asarray(tr.promoter_basal_prob)
    promoter_to_TU = np.asarray(tr.promoter_to_TU)

    summed = np.zeros_like(basal)
    np.add.at(summed, promoter_to_TU, promoter_basal)
    touched = np.zeros(len(basal), dtype=bool)
    touched[promoter_to_TU] = True
    assert np.allclose(summed[touched], basal[touched], rtol=1e-10, atol=0)

    shares = collections.defaultdict(float)
    for p, tu in enumerate(promoter_to_TU):
        shares[int(tu)] += tr.promoter_initiation_share[p]
    assert all(abs(v - 1.0) < 1e-9 for v in shares.values())
