"""Integrity of the committed model-side omics fixtures.

Two properties worth pinning, both of which failed silently somewhere upstream
before these existed:

* **The id key carries the same values as the symbol key.** ``by_id`` was added
  as a re-key of ``by_symbol``, not a re-bake, so the numbers must be identical.
* **The fixtures describe the same ensemble.** A derived vector decoupled from
  its run is how a provenance string drifts from the artifact it describes —
  a downstream card once rendered "104 cells" over values from this 20-cell
  bake. The transcriptome is derived from the pinned reference and the proteome
  from the sweep, so nothing but a test keeps them describing one run.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURES = REPO / "tests/fixtures/population_phenotype_basal"
REFERENCE = REPO / "tests/fixtures/population_phenotype_basal_reference.json"


def _load(name):
    p = FIXTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not baked")
    return json.load(open(p, encoding="utf-8"))


def test_proteome_by_id_preserves_by_symbol_values():
    pro = _load("model_proteome.json")
    assert "by_id" in pro, "proteome carries no id key; cross-source joins have none"
    assert pro["id_key"] == "EcoCyc monomer id"
    # Same values, new key — the re-key must not have moved a single number.
    assert sorted(pro["by_id"].values()) == sorted(pro["by_symbol"].values())
    assert len(pro["by_id"]) == pro["keying"]["n_by_id"]


def test_proteome_id_key_is_not_a_symbol():
    """Symbols are not safe join keys across sources; ids must be actual ids."""
    pro = _load("model_proteome.json")
    sym = set(pro["by_symbol"])
    # A monomer id space and a gene-symbol space should barely intersect.
    overlap = len(set(pro["by_id"]) & sym)
    assert overlap < 0.01 * len(pro["by_id"]), (
        f"{overlap} 'ids' are also symbols — by_id looks symbol-keyed")


def test_transcriptome_is_keyed_by_bare_ecocyc_gene_id():
    tx = _load("model_transcriptome.json")
    assert tx["id_key"] == "EcoCyc gene id"
    keys = list(tx["by_gene_id"])
    assert keys, "empty transcriptome"
    # Gene ids, not cistron ids: the "_RNA" suffix must be gone, and gone
    # because it was read from cistron_data["gene_id"], not string-stripped.
    assert not any(k.endswith("_RNA") for k in keys)
    assert tx["keying"]["n_unmapped"] == 0
    assert tx["keying"]["n_collisions"] == 0


def test_transcriptome_and_proteome_describe_the_same_ensemble():
    """The drift guard. These two fixtures are derived by different paths (the
    transcriptome from the pinned reference, the proteome from the sweep), so
    only this test stops them describing different runs."""
    tx, pro = _load("model_transcriptome.json"), _load("model_proteome.json")
    assert tx["n_cells"] == pro["n_cells"], (
        f"transcriptome says {tx['n_cells']} cells, proteome says "
        f"{pro['n_cells']} — the two committed vectors are from different "
        f"ensembles and any card rendering both would misstate one")
    assert tx["gen_lb"] == pro["gen_lb"]


def test_transcriptome_stamp_matches_its_stated_source():
    """The stamped n_cells must equal the ensemble the vector actually came
    from — not whichever sweep happened to be on disk at bake time."""
    tx = _load("model_transcriptome.json")
    src = tx["source"]
    if src["kind"] != "pinned_reference":
        pytest.skip("transcriptome was baked from a sweep, not the reference")
    ref = json.load(open(REFERENCE, encoding="utf-8"))
    ensemble = ref["stimulus"]["ensemble"]
    assert src["ensemble"] == ensemble
    assert f"{tx['n_cells']} cells" in ensemble, (
        f"stamped n_cells={tx['n_cells']} is not the ensemble the reference "
        f"describes ({ensemble!r})")
    # Width must match the reference vector it was keyed from.
    crit = ref["axes"]["omics.transcriptome"]["criterion"]
    assert tx["keying"]["n_cistrons"] == len(crit["ref_vector"])
