"""Private-payload new-gene overlay: bundle chaining + complexation join.

Covers the three-link path that lets a private payload contribute a strain's
new-gene flat inputs to a v2ecoli ParCa build:

1. ``SourceBundle`` overrides are a CHAIN — v2ecoli's own overrides first, then
   the caller's — so naming a private overlay ADDS keys rather than replacing
   v2ecoli's diverged flat files.
2. ``KnowledgeBaseEcoli`` joins a new-gene ``complexation_reactions.tsv`` when
   the insertion ships one, so a pathway whose enzymes act as protein complexes
   can form them; and does NOT require one, so existing insertions still build.
3. ``InitializeStep`` forwards ``bundle_overrides`` and ``new_genes``, which
   were previously declared and never read.

Everything here is hermetic and public: the new-gene fixture is a copy of the
public ``gfp`` insertion plus one synthetic file. No private payload is needed
and none is referenced.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
from v2ecoli.processes.parca.steps import step_01_initialize as s1

pytestmark = pytest.mark.fast

# The public insertion used as the fixture base. ``template`` is a stub whose
# insertion_location.tsv does not describe a single contiguous insertion, so it
# cannot build; ``gfp`` is the working public example.
BASE_INSERTION = "gfp"
GFP_MONOMER = "NG-GFP-MONOMER"


def _bundle_paths(prefix):
    """{filename: source Path} for every bundle key under ``prefix``."""
    index = SourceBundle()._index
    return {Path(p).name: Path(p) for k, p in index.items() if k.startswith(prefix)}


def _write_overlay(root: Path, subdir: str, extra: dict[str, str] | None = None):
    """Copy the public insertion to ``root`` as ``subdir``, plus ``extra`` files.

    Returns the overlay manifest path. Layout mirrors the public bundle's:
    ``source_path`` is resolved relative to the manifest's own directory, so the
    rows read ``flat/…`` exactly as the public reference bundle's do.
    """
    dest = root / "flat" / "new_gene_data" / subdir
    dest.mkdir(parents=True)
    for name, src in _bundle_paths(f"new_gene_data__{BASE_INSERTION}__").items():
        shutil.copy(src, dest / name)
    for name, text in (extra or {}).items():
        (dest / name).write_text(text)

    manifest = root / "reference_bundle_overlay.tsv"
    rows = ["\t".join(["canonical_key", "source_path", "description", "schema_name"])]
    for f in sorted(dest.glob("*.tsv")):
        rel = f"new_gene_data/{subdir}/{f.name}"
        rows.append("\t".join(
            [rel[: -len(".tsv")].replace("/", "__"), f"flat/{rel}", "test fixture", ""]
        ))
    manifest.write_text("\n".join(rows) + "\n")
    return manifest


# A synthetic product metabolite. Column set must match the base metabolites
# table exactly or ``_join_data`` refuses the join.
PRODUCT_METABOLITE = (
    '"id"\t"common_name"\t"synonyms"\t"chemical_formula"\t"mw"'
    '\t"molecular_charge"\t"_smiles"\n'
    '"NG-TEST-PRODUCT"\t"test product"\t[]\t"C10H10N2O2"\t190.2\t0\t""\n'
)

# A homodimer of the public insertion's monomer. Column set must match the base
# complexation_reactions table exactly or ``_join_data`` refuses the join.
HOMODIMER = (
    '"id"\t"stoichiometry"\t"common_name"\t"cofactors"\n'
    '"NG-TEST-CPLX_RXN"\t{"NG-TEST-CPLX": 1, "' + GFP_MONOMER + '": -2}'
    '\t"test homodimer"\t{}\n'
)


# --------------------------------------------------------------------------
# 1. The override chain
# --------------------------------------------------------------------------

def _diverged_keys(bundle):
    """Keys resolving to v2ecoli's own flat_overrides rather than upstream."""
    return {k for k, p in bundle._index.items() if "flat_overrides" in str(p)}


def test_caller_overrides_do_not_revert_v2ecolis_own_flat_files(tmp_path):
    """The regression this chain exists to prevent.

    When ``overrides`` replaced rather than extended the defaults, passing any
    caller manifest silently reverted v2ecoli's diverged flat files to their
    upstream ecoli-sources versions -- no warning, validation still passing, and
    the ParCa quietly fitting different biology.
    """
    manifest = _write_overlay(tmp_path, "overlay_probe")
    default = SourceBundle()
    composed = SourceBundle(overrides=manifest)

    assert _diverged_keys(default), "fixture assumes v2ecoli has diverged files"
    assert _diverged_keys(composed) == _diverged_keys(default)


def test_overlay_adds_keys_without_displacing_the_base(tmp_path):
    manifest = _write_overlay(tmp_path, "overlay_probe")
    default = SourceBundle()
    composed = SourceBundle(overrides=manifest)

    added = set(composed._index) - set(default._index)
    assert added, "the overlay contributed no keys"
    assert all(k.startswith("new_gene_data__overlay_probe__") for k in added)
    assert not set(default._index) - set(composed._index), "a base key was dropped"


def test_overrides_accepts_a_list_and_applies_in_order(tmp_path):
    one = _write_overlay(tmp_path / "a", "probe_a")
    two = _write_overlay(tmp_path / "b", "probe_b")
    composed = SourceBundle(overrides=[one, two])

    assert [p.name for p in composed.override_chain][0] == "parca_overrides.tsv"
    assert len(composed.override_chain) == 3
    for subdir in ("probe_a", "probe_b"):
        assert any(k.startswith(f"new_gene_data__{subdir}__") for k in composed._index)


# --------------------------------------------------------------------------
# 2. The complexation join
# --------------------------------------------------------------------------

def _kb(bundle, new_genes):
    return KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False, remove_rrff=False,
        stable_rrna=False, new_genes_option=new_genes, bundle=bundle,
    )


def _new_gene_complexes(kb):
    return sorted(
        c["id"] for c in kb.complexation_reactions if str(c["id"]).startswith("NG-")
    )


def test_new_gene_complexation_is_joined_when_the_insertion_ships_one(tmp_path):
    """Without this the complex is never formed: its monomers accumulate with
    nothing to do, and any consumer looking up the complex as a catalyst fails
    on an id the bulk store does not contain."""
    manifest = _write_overlay(
        tmp_path, "gfp_cplx", extra={"complexation_reactions.tsv": HOMODIMER}
    )
    kb = _kb(SourceBundle(overrides=manifest), "gfp_cplx")

    assert _new_gene_complexes(kb) == ["NG-TEST-CPLX_RXN"]


def test_new_gene_complexation_absent_still_builds(tmp_path):
    """The file is OPTIONAL. Requiring it -- i.e. adding it to the asserted
    ``new_gene_shared_files`` list -- would break every existing insertion,
    since neither public ``gfp`` nor ``template`` ships one."""
    manifest = _write_overlay(tmp_path, "gfp_plain")
    kb = _kb(SourceBundle(overrides=manifest), "gfp_plain")

    assert _new_gene_complexes(kb) == []
    assert any(str(p["id"]).startswith("NG-") for p in kb.proteins)


def test_new_gene_metabolites_are_joined_and_reach_the_bulk_store(tmp_path):
    """A heterologous pathway's product needs a bulk entry to accumulate into.

    Without this the pathway can be fully built -- genes expressed, enzymes
    synthesised and complexed -- and still have nowhere to put what it makes,
    so no product count and no product KPI. Measured on a real heterologous
    pathway insertion: with the file joined, the pathway's product and its
    intermediates appear in the bulk store; without it they are absent
    entirely, while every other part of the strain builds correctly.
    """
    from v2ecoli.processes.parca.reconstruction.ecoli.simulation_data import (
        SimulationDataEcoli,
    )

    manifest = _write_overlay(
        tmp_path, "gfp_product", extra={"metabolites.tsv": PRODUCT_METABOLITE}
    )
    kb = _kb(SourceBundle(overrides=manifest), "gfp_product")
    assert "NG-TEST-PRODUCT" in [m["id"] for m in kb.metabolites]

    sim_data = SimulationDataEcoli()
    sim_data.initialize(raw_data=kb)
    bulk_ids = set(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    assert "NG-TEST-PRODUCT[c]" in bulk_ids


def test_new_gene_metabolites_colliding_with_the_host_are_refused(tmp_path):
    """A payload re-declaring a HOST molecule must fail loudly, not silently win.

    Joined rows do not merge: both survive, and every consumer that builds an
    id-keyed dict (molecular weights, charges) takes the last one. A
    heterologous pathway consumes host metabolites, so its own tables can
    plausibly name one -- ``TRP`` is the worked example, being both a base-table
    metabolite and the substrate of a real pathway we build. Silently rewriting
    tryptophan's mass and charge would corrupt the host's chemistry with no
    error anywhere.
    """
    collide = (
        '"id"\t"common_name"\t"synonyms"\t"chemical_formula"\t"mw"'
        '\t"molecular_charge"\t"_smiles"\n'
        '"TRP"\t"not really tryptophan"\t[]\t"C1H1"\t1.0\t0\t""\n'
    )
    manifest = _write_overlay(
        tmp_path, "gfp_collide", extra={"metabolites.tsv": collide}
    )
    with pytest.raises(ValueError, match="already exist in the base table"):
        _kb(SourceBundle(overrides=manifest), "gfp_collide")


def test_new_gene_metabolites_absent_still_builds(tmp_path):
    """Optional, exactly like complexation: neither public insertion ships one."""
    manifest = _write_overlay(tmp_path, "gfp_no_product")
    kb = _kb(SourceBundle(overrides=manifest), "gfp_no_product")

    assert not [m for m in kb.metabolites if str(m["id"]).startswith("NG-")]


def test_public_gfp_insertion_is_unaffected():
    """Regression guard on the shipped public path."""
    baseline = len(_kb(SourceBundle(), "off").complexation_reactions)
    kb = _kb(SourceBundle(), BASE_INSERTION)

    assert _new_gene_complexes(kb) == []
    assert len(kb.complexation_reactions) == baseline


# --------------------------------------------------------------------------
# 3. InitializeStep forwards the two previously-inert fields
# --------------------------------------------------------------------------

def test_initialize_step_forwards_new_genes_and_bundle_overrides():
    """Both were declared composite params that nothing read, so a study could
    name a private overlay or a new-gene insertion and be silently ignored."""
    config = {
        "raw_data": None,
        "bundle_overrides": "/somewhere/overlay.tsv",
        "new_genes": "some_pathway_MG1655_v2",
    }
    with patch.object(s1, "KnowledgeBaseEcoli") as mock_kb, \
            patch.object(s1, "SourceBundle") as mock_bundle:
        mock_kb.return_value = MagicMock()
        s1._resolve_raw_data(config)

    # A list, not the bare string: the CLI records repeatable overrides
    # ';'-joined, so this field is always split into a chain (see
    # test_bundle_overrides_round_trips_multiple_manifests).
    assert mock_bundle.call_args.kwargs["overrides"] == ["/somewhere/overlay.tsv"]
    assert mock_kb.call_args.kwargs["new_genes_option"] == "some_pathway_MG1655_v2"


def _check_genotype(declared_new_genes, raw_new_genes):
    """Drive ``_check_declared_genotype`` the way tests/test_parca_genotype_declaration
    does -- the check is a Step method, and ``_resolve_raw_data`` returns early
    for an injected KB so it never reaches it."""
    kb = MagicMock()
    kb.new_genes_option = raw_new_genes
    kb._bundle = None
    step = object.__new__(s1.InitializeStep)
    step.config = {"bundle_manifest": "", "new_genes": declared_new_genes,
                   "raw_data": kb}
    step._check_declared_genotype()


def test_declared_new_genes_disagreeing_with_raw_data_warns():
    """new_genes changes the GENOME, so a mismatch is not a provenance nit.

    On the injected-raw_data path this step never builds the KB, so without
    this check a config declaring an insertion against a wild-type KB fits WT,
    warns nothing, and records a genotype it does not have.
    """
    with pytest.warns(UserWarning, match="new_genes"):
        _check_genotype("some_pathway_MG1655_v2", "off")


def test_declared_new_genes_agreeing_is_silent():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        _check_genotype("some_pathway_MG1655_v2", "some_pathway_MG1655_v2")


def test_bundle_overrides_round_trips_multiple_manifests(tmp_path):
    """The CLI records repeatable --bundle-overrides as a ';'-joined string.

    Handing that value straight to SourceBundle treats it as ONE path, so it
    round-trips for a single override and fails on two -- and the CLI field is
    provenance-only, so the breakage surfaces only where the value is resolved,
    far from where it was recorded.
    """
    one = _write_overlay(tmp_path / "a", "probe_a")
    two = _write_overlay(tmp_path / "b", "probe_b")
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(s1, "KnowledgeBaseEcoli", return_value=MagicMock()), \
            patch.object(s1, "SourceBundle", side_effect=_capture):
        s1._resolve_raw_data({"raw_data": None,
                              "bundle_overrides": f"{one};{two}"})

    assert captured["overrides"] == [str(one), str(two)]


def test_initialize_step_defaults_new_genes_off():
    with patch.object(s1, "KnowledgeBaseEcoli") as mock_kb, \
            patch.object(s1, "SourceBundle"):
        mock_kb.return_value = MagicMock()
        s1._resolve_raw_data({"raw_data": None})

    assert mock_kb.call_args.kwargs["new_genes_option"] == "off"
