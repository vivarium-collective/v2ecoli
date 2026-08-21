"""Tests for v2ecoli.perturbations.new_gene_cache — the driver that gives
``set_new_gene_expression`` a caller.

Hermetic by default. The one thing that genuinely cannot be tested hermetically
is a real cache round trip: there is no public, committable new-gene sim_data on
this machine (the shipped ``models/parca/parca_state.pkl.gz`` is a basal fit with
no ``is_new_gene`` cistrons), and producing one needs a full ParCa build whose
output is too large to commit. So the round trip is split — an ORDERING test that
runs everywhere, plus an env-gated end-to-end test that skips by default. See
``test_round_trip_builds_a_composite_from_the_saved_cache`` for what the gate
buys and, more importantly, for what the hermetic half does NOT cover.

The fake here is defined with module-level classes rather than
``type("T", (), {})()`` (the style of ``test_new_gene_perturbation.py``) because
the driver deep-copies its input with ``pickle`` and dynamically created classes
are not picklable.
"""
import os

import numpy as np
import pytest

from v2ecoli.perturbations import build_new_gene_cache
from v2ecoli.perturbations.translation import translation_efficiency_override


class _Struct:
    def __init__(self, arr):
        self.struct_array = arr


class _Transcription:
    pass


class _Translation:
    pass


class _Process:
    pass


class _FakeSimData:
    """Two native genes plus ``n_new`` new genes, with the four arrays the real
    ``adjust_new_gene_final_expression`` writes to.

    Implements the reference's semantics (``reconstruction/ecoli/
    simulation_data.py``): assign every target from its baseline, then
    renormalize ``rna_expression`` / ``exp_free`` / ``exp_ppgpp`` ONCE.
    """

    def __init__(self, n_new=2, baseline=1e-4):
        layout = ["EG10001_RNA", "EG10002_RNA"] + [f"NG-GFP{i}" for i in range(n_new)]
        is_new = [g.startswith("NG") for g in layout]

        self.process = _Process()
        self.process.transcription = _Transcription()
        self.process.translation = _Translation()

        self.process.transcription.cistron_data = _Struct(np.array(
            list(zip(layout, is_new)), dtype=[("id", "U32"), ("is_new_gene", "?")]))
        self.process.translation.monomer_data = _Struct(np.array(
            [(g, f"{g}-MONOMER[c]") for g in layout],
            dtype=[("cistron_id", "U32"), ("id", "U40")]))
        self.process.transcription.rna_data = {
            "id": np.array([f"{g}[c]" for g in layout], dtype="U40")}

        native = np.array([0.0 if n else 0.5 for n in is_new])
        self.process.transcription.rna_expression = {"basal": native.copy()}
        self.process.transcription.exp_free = native.copy()
        self.process.transcription.exp_ppgpp = native.copy()
        # NOT all ones: with a baseline of 1.0 "assign" and "multiply into the
        # existing value" are numerically identical and prove nothing.
        self.process.translation.translation_efficiencies_by_monomer = np.array(
            [1.0, 1.0] + [2.0 * (i + 1) for i in range(n_new)])
        self._baseline = baseline

    def adjust_new_gene_final_expression(self, indices, factors):
        tx = self.process.transcription
        for i, f in zip(indices, factors):
            for exp in tx.rna_expression.values():
                exp[i] = self._baseline * f
            tx.exp_free[i] = self._baseline * f
            tx.exp_ppgpp[i] = self._baseline * f
        for exp in tx.rna_expression.values():
            exp /= exp.sum()
        tx.exp_free /= tx.exp_free.sum()
        tx.exp_ppgpp /= tx.exp_ppgpp.sum()


def _spy_on_save(monkeypatch, sink):
    """Replace ``save_sim_input`` with a spy that snapshots sim_data AS SAVED.

    The snapshot is the point: asserting only "save was called last" is vacuous
    — it passes against a driver that saves an unperturbed sim_data. What must
    be true is that the values are ALREADY IN the arrays at the moment the
    bundle is extracted from them.
    """
    def _save(sim_data, bundle_dir="out/cache", seed=0, condition=None,
              fixed_media=None, **kwargs):
        tx = sim_data.process.transcription
        sink.append({
            "bundle_dir": bundle_dir,
            "seed": seed,
            "condition": condition,
            "fixed_media": fixed_media,
            "exp_free": np.array(tx.exp_free, copy=True),
            "exp_ppgpp": np.array(tx.exp_ppgpp, copy=True),
            "rna_expression": np.array(tx.rna_expression["basal"], copy=True),
            "te": np.array(
                sim_data.process.translation.translation_efficiencies_by_monomer,
                copy=True),
        })

    monkeypatch.setattr("v2ecoli.core.save_sim_input", _save)
    return sink


# --------------------------------------------------------------------------- #
# Ordering — the hermetic half of the round trip.
# --------------------------------------------------------------------------- #

def test_new_gene_values_are_in_sim_data_when_the_bundle_is_saved(monkeypatch, tmp_path):
    # Catches: applying the perturbation AFTER save_sim_input; saving the
    # untouched original instead of the perturbed deep copy (a bug the deep copy
    # itself makes possible); and any future refactor that extracts the bundle
    # from a different object than the one that was modified. All three produce
    # a cache that is a pre-perturbation cache wearing a perturbed cache's name.
    sd = _FakeSimData(n_new=2)
    saved = _spy_on_save(monkeypatch, [])
    cache_dir = str(tmp_path / "cache-new-genes")

    result = build_new_gene_cache(
        sd, cache_dir, expression=1e6, translation_efficiency=0.75,
        rel_exp_adj=[1.0, 3.0], rel_trl_eff_adj=[1.0, 2.0],
        seed=7, condition="acetate", fixed_media="minimal_acetate")

    assert len(saved) == 1, "the driver must save exactly one bundle"
    snap = saved[0]
    assert snap["bundle_dir"] == cache_dir
    assert (snap["seed"], snap["condition"], snap["fixed_media"]) == (
        7, "acetate", "minimal_acetate")

    ng_rna = result["applied"]["rna_indices"]
    ng_mon = result["applied"]["monomer_indices"]

    # Translation efficiency: the assigned values are in the array as saved.
    assert [snap["te"][i] for i in ng_mon] == pytest.approx([0.75, 1.5])
    # Native monomers were not disturbed.
    assert snap["te"][0] == 1.0 and snap["te"][1] == 1.0

    # Expression: non-zero (they start at exactly 0) and in the 1:3 ratio asked
    # for, in every array the reference renormalizes.
    for key in ("exp_free", "exp_ppgpp", "rna_expression"):
        vals = [snap[key][i] for i in ng_rna]
        assert all(v > 0 for v in vals), f"{key} still silent at save time"
        assert vals[1] / vals[0] == pytest.approx(3.0), key


def test_the_callers_sim_data_is_not_mutated(monkeypatch, tmp_path):
    # Catches: mutating in place. set_new_gene_expression modifies sim_data
    # directly, so without the deep copy a grid loop over one loaded sim_data
    # would silently carry grid point k-1's induction into grid point k.
    monkeypatch.setattr("v2ecoli.core.save_sim_input", lambda *a, **k: None)
    sd = _FakeSimData(n_new=2)
    before = {
        "exp_free": np.array(sd.process.transcription.exp_free, copy=True),
        "exp_ppgpp": np.array(sd.process.transcription.exp_ppgpp, copy=True),
        "rna_expression": np.array(
            sd.process.transcription.rna_expression["basal"], copy=True),
        "te": np.array(
            sd.process.translation.translation_efficiencies_by_monomer, copy=True),
    }

    build_new_gene_cache(sd, str(tmp_path / "c"), expression=1e6,
                         translation_efficiency=0.75)

    assert np.array_equal(sd.process.transcription.exp_free, before["exp_free"])
    assert np.array_equal(sd.process.transcription.exp_ppgpp, before["exp_ppgpp"])
    assert np.array_equal(sd.process.transcription.rna_expression["basal"],
                          before["rna_expression"])
    assert np.array_equal(
        sd.process.translation.translation_efficiencies_by_monomer, before["te"])


def test_grid_points_built_from_one_sim_data_do_not_contaminate_each_other(
        monkeypatch, tmp_path):
    # The consequence the previous test exists to prevent, asserted directly:
    # a second grid point off a reused sim_data must equal the same grid point
    # built from a fresh one. Without isolation the renormalization from the
    # first (high) induction leaves the natives crushed, and the second point
    # lands somewhere else entirely.
    saved = _spy_on_save(monkeypatch, [])
    reused = _FakeSimData(n_new=2)
    build_new_gene_cache(reused, str(tmp_path / "a"), expression=1e8,
                         translation_efficiency=1.0)
    build_new_gene_cache(reused, str(tmp_path / "b"), expression=1e2,
                         translation_efficiency=1.0)
    build_new_gene_cache(_FakeSimData(n_new=2), str(tmp_path / "c"),
                         expression=1e2, translation_efficiency=1.0)

    assert np.allclose(saved[1]["exp_free"], saved[2]["exp_free"])
    assert np.allclose(saved[1]["rna_expression"], saved[2]["rna_expression"])


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #

def test_provenance_records_the_as_assigned_values_not_the_cached_ones(
        monkeypatch, tmp_path):
    # Catches: recording cache-relative numbers. get_polypeptide_initiation_config
    # stores normalize(translation_efficiencies_by_monomer)
    # (v2ecoli/library/sim_data.py:1051), so the cached value at a new-gene
    # monomer depends on every other monomer and moves whenever any of them
    # moves. Recording that as "what was applied" would be a number whose
    # meaning is not the perturbation. The record must be what was asked for.
    from v2ecoli.library.fitting import normalize

    saved = _spy_on_save(monkeypatch, [])
    result = build_new_gene_cache(
        _FakeSimData(n_new=2), str(tmp_path / "c"),
        expression=1e6, translation_efficiency=0.5,
        rel_exp_adj=[1.0, 3.0], rel_trl_eff_adj=[2.0, 1.0])

    assert result["cache_dir"] == str(tmp_path / "c")
    applied = result["applied"]
    assert applied["expression_factors"] == pytest.approx([1e6, 3e6])
    assert applied["translation_efficiencies"] == pytest.approx([1.0, 0.5])

    # As assigned == what is in sim_data ...
    te = saved[0]["te"]
    assert [te[i] for i in applied["monomer_indices"]] == pytest.approx(
        applied["translation_efficiencies"])
    # ... and demonstrably NOT what the consumer will read out of the cache.
    cached = normalize(te)
    assert [cached[i] for i in applied["monomer_indices"]] != pytest.approx(
        applied["translation_efficiencies"], rel=1e-6)


def test_a_sim_data_without_new_genes_fails_fast(monkeypatch, tmp_path):
    # Catches: writing a cache that looks perturbed but is not. The shipped
    # basal fixture is exactly this case, so the failure has to come before any
    # bundle is written, not after.
    saved = _spy_on_save(monkeypatch, [])
    with pytest.raises(ValueError, match="no new-gene cistrons"):
        build_new_gene_cache(_FakeSimData(n_new=0), str(tmp_path / "c"),
                             expression=1e6, translation_efficiency=1.0)
    assert saved == [], "no cache may be written when the perturbation failed"


# --------------------------------------------------------------------------- #
# The translation-efficiency composition trap (new_genes.py:47-54).
# --------------------------------------------------------------------------- #

_MONOMERS = ["EG10001-MONOMER[c]", "EG10002-MONOMER[c]", "NG-GFP0-MONOMER[c]"]
_GENES = ["EG10001", "EG10002", "NG-GFP0"]
_NG = 2  # index of the new-gene monomer in both lists


def _bundle(efficiencies):
    return {"configs": {
        "ecoli-polypeptide-initiation": {
            "monomer_ids": list(_MONOMERS),
            "translation_efficiencies": np.asarray(efficiencies, dtype=float),
            "monomer_index_to_cistron_index": {i: i for i in range(len(_MONOMERS))},
        },
        "rna_synth_prob_listener": {"gene_ids": list(_GENES)},
    }}


def test_native_override_preserves_new_gene_efficiencies_from_the_same_cache():
    # translation_efficiency_override returns a FULL REPLACEMENT array, applied
    # by baseline's override seam as one assignment. Composing a native
    # knockout against the driver's OWN cache is therefore safe: the new-gene
    # entry is already in the array the replacement is built from.
    # Catches: a change that stops the driver's values reaching the bundle the
    # override reads, or an override that rebuilds efficiencies from some other
    # source instead of the bundle it was handed.
    driver_cache = _bundle([1.0, 1.0, 0.75])
    override = translation_efficiency_override(driver_cache, {"EG10001": 0.0})
    patched = override["ecoli-polypeptide-initiation.translation_efficiencies"]

    assert patched[0] == 0.0, "the native knockout must still apply"
    assert patched[_NG] == pytest.approx(0.75), (
        "new-gene efficiency discarded by a same-cache override")


def test_native_override_from_a_pre_modification_cache_discards_them():
    # The trap itself, pinned. Same knockout, computed against the cache as it
    # was BEFORE the new genes were induced: the whole-array replacement carries
    # that cache's silent new-gene entry, so applying it to the driver's cache
    # wipes the induction — with no error, and with the arm still differing in
    # its inputs. This is why the driver applies the perturbation before the
    # cache is built rather than composing two overrides afterwards.
    # Catches: the trap becoming untrue in a way the docs still claim (e.g. a
    # sparse override), and vice versa — it fails if either half changes.
    pre_cache = _bundle([1.0, 1.0, 0.0])
    override = translation_efficiency_override(pre_cache, {"EG10001": 0.0})
    patched = override["ecoli-polypeptide-initiation.translation_efficiencies"]

    assert patched[0] == 0.0
    # The replacement really is the pre-modification array, not an empty one:
    # every untargeted entry is carried across verbatim.
    assert patched[1] == pytest.approx(1.0)
    assert patched[_NG] == 0.0, (
        "the pre-modification cache's silent new-gene entry must be what a "
        "cross-cache override carries — that is the discard being described")
    assert patched[_NG] != pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Real round trip — env-gated, skipped by default.
# --------------------------------------------------------------------------- #

_ROUND_TRIP_STATE = os.environ.get("V2ECOLI_NEW_GENE_CACHE")


@pytest.mark.skipif(
    not _ROUND_TRIP_STATE,
    reason="set V2ECOLI_NEW_GENE_CACHE=/path/to/parca_state.pkl[.gz] from a "
           "`v2ecoli-parca --new-genes ...` build to run the real round trip")
def test_round_trip_builds_a_composite_from_the_saved_cache(tmp_path):
    """The end-to-end claim the hermetic tests above deliberately do NOT make.

    Everything above stops at the sim_data boundary: they prove the right values
    are in sim_data when the bundle is extracted. They do not prove the bundle
    reloads, that ``baseline`` builds from it, or that the induction survives
    into the built composite's process configs. That needs a real new-gene
    sim_data, which cannot be committed — hence the gate.
    """
    from v2ecoli.composites.ecoli_baseline import baseline
    from v2ecoli.core import build_core, load_cache_bundle
    from v2ecoli.perturbations import new_gene_indices
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)

    sim_data = hydrate_sim_data_from_state(load_parca_state(_ROUND_TRIP_STATE))
    _, _, _, monomer_indices = new_gene_indices(sim_data)
    # Unequal weights, so the surviving ratios are informative. Absolute values
    # do not survive the cache's L1 normalization; ratios do.
    weights = [float(i + 1) for i in range(len(monomer_indices))]

    cache_dir = str(tmp_path / "cache-new-genes")
    result = build_new_gene_cache(
        sim_data, cache_dir, expression=1e6, translation_efficiency=1.0,
        rel_trl_eff_adj=weights)

    bundle = load_cache_bundle(cache_dir)
    cached_te = np.asarray(
        bundle["configs"]["ecoli-polypeptide-initiation"]["translation_efficiencies"])
    ng = [cached_te[i] for i in result["applied"]["monomer_indices"]]
    assert all(v > 0 for v in ng), "new genes still silent in the reloaded cache"
    for i, w in enumerate(weights):
        assert ng[i] / ng[0] == pytest.approx(w / weights[0], rel=1e-6)

    doc = baseline(core=build_core(), seed=0, cache_dir=cache_dir)
    instance = doc["state"]["agents"]["0"]["ecoli-polypeptide-initiation"]["instance"]
    built = np.asarray(instance.parameters["translation_efficiencies"])
    for i, idx in enumerate(result["applied"]["monomer_indices"]):
        assert built[idx] == pytest.approx(cached_te[idx])
        assert built[idx] / built[result["applied"]["monomer_indices"][0]] == (
            pytest.approx(weights[i] / weights[0], rel=1e-6))
