"""Regression: the ppGpp transcription program must not amplify *null* genes.

Two floating-point knife-edges let a gene with ~0 expression capture a large
share of transcription whenever its ppGpp-adjusted basal underflowed to exactly
0.0 (numpy/BLAS-build dependent):

  1. ParCa ``adjust_ppgpp_expression_for_tfs`` divided ``new_prob / old_prob``;
     for a null gene ``old_prob`` is at the numerical-noise floor, so the ratio
     explodes and (after renormalization) that gene steals expression mass.
  2. The runtime ``transcript_initiation`` ppGpp branch used
     ``ppgpp_scale[ppgpp_scale == 0] = 1``, restoring the TF delta at full
     strength wherever the ppGpp basal hit exactly 0.0 — so a null gene's delta
     dominated the (renormalized) initiation probabilities.

Together they drove a single transcription unit to ~19% of all initiation on
carbon-poor (acetate) media in v2ecoli while vEcoli stayed at ~0, an
irreproducible divergence that depended only on whether a denormal flushed to
zero. Both are fixed by suppressing null genes (scale the delta by the real,
near-zero basal; leave floor-level genes unscaled in the ParCa fit).

This test locks the runtime behavior in: no single TU may capture an
implausible share of the ppGpp synthesis-probability distribution. A real rRNA
operon tops out around 3%; the pre-fix knife-edge produced ~19%. The 8% ceiling
sits cleanly between the two, so a revert of either fix trips it.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
# A real rRNA operon peaks ~3% of the synth-prob distribution; the pre-fix
# null-gene knife-edge produced ~19%. 8% separates legitimate from spurious.
MAX_SINGLE_TU_SYNTH_PROB = 0.08
# Carbon-poor media is where the knife-edge bit hardest (strong TF regulation,
# many genes whose ppGpp basal underflows to exactly 0).
_FIXTURE = REPO / "models" / "parca" / "parca_state.pkl.gz"

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not _FIXTURE.exists() and not os.environ.get("V2_ACETATE_CACHE"),
        reason="needs the committed ParCa fixture (or V2_ACETATE_CACHE) to "
               "build an acetate composite",
    ),
]


@pytest.fixture(scope="module")
def acetate_cache(tmp_path_factory):
    """An acetate-condition composite cache built from the committed ParCa
    fixture. Honors ``V2_ACETATE_CACHE`` (e.g. a prebuilt
    ``out/cache_acetate_regularized``) to skip the ~3 s rebuild."""
    env_cache = os.environ.get("V2_ACETATE_CACHE")
    if env_cache and (Path(env_cache) / "sim_data_cache.dill").exists():
        return env_cache
    cache = tmp_path_factory.mktemp("acetate_cache")
    subprocess.run(
        [sys.executable, "scripts/build_cache.py",
         "--fixture", str(_FIXTURE), "--cache", str(cache),
         "--media-condition", "acetate", "--fixed-media", "minimal_acetate"],
        cwd=str(REPO), check=True)
    return str(cache)


@pytest.mark.timeout(600)
def test_no_null_gene_dominates_ppgpp_synth_prob(acetate_cache):
    """Run an acetate composite and confirm no single transcription unit
    captures an implausible share of the ppGpp synthesis-probability
    distribution (the pre-fix null-gene amplification signature)."""
    import v2ecoli.processes.transcript_initiation as TI

    captured = []
    orig = TI.TranscriptInitiation.update

    def spy(self, *a, **k):
        result = orig(self, *a, **k)
        synth = (result.get("listeners", {})
                 .get("rna_synth_prob", {})) if isinstance(result, dict) else {}
        tp = synth.get("target_rna_synth_prob")
        if tp is not None and np.ndim(tp) >= 1 and len(np.atleast_1d(tp)) > 1:
            captured.append(np.asarray(tp, dtype=float))
        return result

    TI.TranscriptInitiation.update = spy
    try:
        from v2ecoli import build_composite
        composite = build_composite("baseline", cache_dir=acetate_cache, seed=0)
        composite.run(40)
    finally:
        TI.TranscriptInitiation.update = orig

    assert len(captured) >= 20, (
        f"expected >=20 transcript-initiation steps, got {len(captured)}")
    # Time-average over the settled window (skip the first few warm-up steps).
    mean_synth = np.mean(np.array(captured[10:]), axis=0)
    assert abs(mean_synth.sum() - 1.0) < 1e-6, "synth-prob is not normalized"

    peak = float(mean_synth.max())
    peak_tu = int(mean_synth.argmax())
    assert peak < MAX_SINGLE_TU_SYNTH_PROB, (
        f"a single TU (index {peak_tu}) captured {peak:.1%} of the ppGpp "
        f"synthesis-probability distribution (ceiling {MAX_SINGLE_TU_SYNTH_PROB:.0%}). "
        f"This is the null-gene amplification signature — check the "
        f"transcript_initiation ppgpp_scale guard and the ParCa "
        f"adjust_ppgpp_expression_for_tfs division guard.")
