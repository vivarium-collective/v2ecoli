"""Experimental-transcriptome ParCa path: selector, cross-fill, guards, stamp.

ecoli-sources has shipped the supplier half of this for a long time — the
``rnaseq_experimental_tpms`` canonical key is in the reference bundle, is in
``REQUIRED_CANONICAL_KEYS``, and is what ``processing.genotypes.knockdown()``
rewrites. The consumer half was ported to v2ecoli as ``library/ingestion.py``,
never imported by anything, and deleted as dead code in #288. Both halves were
individually defensible; together they made a knockdown variant bundle validate,
hash stably, build successfully and change nothing.

What is covered here:

1. ``KnowledgeBaseEcoli`` loads the long-form TPM tier BY CANONICAL KEY onto a
   fixed attribute — not through ``list_of_dict_filenames``, which would name
   the attribute after the variant's filename and lose it again.
2. ``sim_data.rnaseq_source`` selects the tier, and an experimental dataset
   actually reaches expression.
3. Cross-fill fills exactly the genes the experimental dataset omits, from the
   ``rnaseq_basal_tpms`` tier (ecoli-sources' declared role for it), and can be
   turned off — with the second-order effect on RNA-seq coverage asserted, not
   just the zero-fill.
4. All four guards fire, and the provenance stamp makes two builds
   distinguishable from the artifact alone.

Everything is hermetic and public: synthetic TPM tables written into
``tmp_path`` and a variant manifest built from the installed public reference
bundle. No private payload, no network.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.process import (
    transcription as tx,
)
from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    RNASEQ_BASAL_KEY,
    RNASEQ_EXPERIMENTAL_KEY,
    KnowledgeBaseEcoli,
    load_tpm_table,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

pytestmark = pytest.mark.fast

WIDE_KEY = "rna_seq_data__rnaseq_rsem_tpm_mean"
DEFAULT_CONDITION = "M9 Glucose minus AAs"


# ---------------------------------------------------------------------------
# Fixtures — a real KB once, and cheap stubs for everything else.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kb():
    """One real KnowledgeBaseEcoli off the default bundle (~2.5 s).

    Module-scoped because the resolver's inputs are read-only; nothing here
    mutates the KB.
    """
    return KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False, remove_rrff=False,
        stable_rrna=False)


@pytest.fixture(scope="module")
def model_gene_ids(kb):
    return [gene["id"] for gene in kb.genes]


class _SimData:
    """The three fields ``resolve_basal_seq_data`` reads off sim_data."""

    def __init__(self, rnaseq_source="reference", rnaseq_cross_fill=True,
                 basal_expression_condition=DEFAULT_CONDITION):
        self.rnaseq_source = rnaseq_source
        self.rnaseq_cross_fill = rnaseq_cross_fill
        self.basal_expression_condition = basal_expression_condition


class _RawData:
    """Stand-in for the parts of KnowledgeBaseEcoli the resolver touches.

    Lets every behavioural case run without a 2.5 s KB build, while the
    ``kb``-backed tests above prove the real KB populates these same fields.
    """

    def __init__(self, wide_rows, tables=None, sources=None, bundle=None):
        self.rna_seq_data = type("_DS", (), {})()
        self.rna_seq_data.rnaseq_rsem_tpm_mean = wide_rows
        self.rnaseq_tpm_tables = tables or {}
        self.rnaseq_tpm_sources = sources or {}
        self._bundle = bundle


def _wide_rows(pairs, condition=DEFAULT_CONDITION):
    return [{"Gene": g, condition: v} for g, v in pairs]


def _tpm_frame(pairs):
    return pd.DataFrame({"gene_id": [g for g, _ in pairs],
                         "tpm_mean": [v for _, v in pairs]})


def _write_tpm(path: Path, pairs):
    _tpm_frame(pairs).to_csv(path, sep="\t", index=False)
    return path


def _variant_bundle(tmp_path: Path, overrides: dict, generated=()):
    """A complete variant manifest overriding ``overrides`` (key -> file path).

    Mirrors what ``ecoli-sources`` ``compose_variant_bundle`` produces: a full
    manifest (never a diff) plus a ``genotype.json`` sidecar naming the keys the
    variant GENERATED. ``source_path`` entries are absolute so no data is copied.
    """
    from ecoli_sources import BUNDLE_PATH

    base = Path(BUNDLE_PATH)
    df = pd.read_csv(base, sep="\t", comment="#")
    df["source_path"] = [
        str((base.parent / str(p)).resolve()) for p in df["source_path"]]
    for key, path in overrides.items():
        df.loc[df["canonical_key"] == key, "source_path"] = str(Path(path).resolve())

    out = tmp_path / "reference_bundle.tsv"
    df.to_csv(out, sep="\t", index=False)
    if generated:
        (tmp_path / "genotype.json").write_text(
            json.dumps({"overridden_keys": sorted(generated)}))
    return out


def _resolve(raw, sim):
    """Run the resolver, returning ``(seq_data, provenance, warning messages)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seq_data, provenance = tx.resolve_basal_seq_data(
            raw, sim, list(_all_gene_ids(raw)))
    return seq_data, provenance, [str(w.message) for w in caught]


def _all_gene_ids(raw):
    return [row["Gene"] for row in raw.rna_seq_data.rnaseq_rsem_tpm_mean]


# ---------------------------------------------------------------------------
# 1. The KB loads the long-form tier by canonical key.
# ---------------------------------------------------------------------------


def test_kb_loads_the_tpm_tier_by_canonical_key(kb):
    """Both RFC-010 tiers land on a FIXED attribute, keyed by canonical key.

    The fixed name is the point. Routing these through ``list_of_dict_filenames``
    would name the attribute after the file's basename, so a knockdown variant
    bundle (whose key points at ``rnaseq_experimental_tpms__kd.tsv``) would land
    on a per-variant attribute nothing reads — the same silent no-op this whole
    change removes.
    """
    assert RNASEQ_EXPERIMENTAL_KEY in kb.rnaseq_tpm_tables
    assert RNASEQ_BASAL_KEY in kb.rnaseq_tpm_tables
    table = kb.rnaseq_tpm_tables[RNASEQ_EXPERIMENTAL_KEY]
    assert {"gene_id", "tpm_mean"} <= set(table.columns)
    assert len(table) > 4000
    assert kb.rnaseq_tpm_sources[RNASEQ_EXPERIMENTAL_KEY].endswith(".tsv")


def test_spreadsheets_reader_cannot_read_this_tier():
    """Why a pandas reader exists at all, pinned as an executable fact.

    The KB's own ``read_tsv`` is a JsonReader — it ``json.loads`` every cell and
    works only because every flat KB file is JSON-quoted. The ecoli-sources TPM
    tier is plain unquoted TSV. If this ever starts passing, ``load_tpm_table``
    can collapse into the normal flat-file path; until then it must not.
    """
    from v2ecoli.processes.parca.reconstruction.spreadsheets import read_tsv
    from ecoli_sources import BUNDLE_PATH

    path = SourceBundle(base_manifest=BUNDLE_PATH).path(RNASEQ_EXPERIMENTAL_KEY)
    with pytest.raises(ValueError, match="failed to parse json string"):
        read_tsv(str(path))
    # ...and the reader this change adds does read it.
    assert len(load_tpm_table(path)) > 4000


# ---------------------------------------------------------------------------
# 2. The selector reaches expression.  (acceptance criterion 2)
# ---------------------------------------------------------------------------


def test_reference_is_the_default_and_reads_the_wide_table():
    """Acceptance criterion 1: the pre-existing path, unchanged.

    Written against the legacy expression literally (dict comprehension over the
    wide rows at ``basal_expression_condition``) rather than against the new
    code, so it stays a regression guard rather than a tautology.
    """
    raw = _RawData(_wide_rows([("EG10001", 5.0), ("EG10002", 7.0)]))
    seq_data, prov, caught = _resolve(raw, _SimData())

    legacy = {r["Gene"]: r[DEFAULT_CONDITION]
              for r in raw.rna_seq_data.rnaseq_rsem_tpm_mean}
    assert seq_data == legacy
    assert prov["source"] == "reference"
    assert caught == []


def test_experimental_dataset_actually_reaches_expression(tmp_path):
    """Acceptance criterion 2: the table wins over the wide reference."""
    raw = _RawData(
        _wide_rows([("EG10001", 5.0), ("EG10002", 7.0)]),
        tables={RNASEQ_EXPERIMENTAL_KEY:
                _tpm_frame([("EG10001", 111.0), ("EG10002", 222.0)])},
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    seq_data, prov, _ = _resolve(raw, _SimData("experimental"))

    assert seq_data == {"EG10001": 111.0, "EG10002": 222.0}
    assert prov["source"] == "experimental"
    assert prov["canonical_key"] == RNASEQ_EXPERIMENTAL_KEY


# ---------------------------------------------------------------------------
# 3. Cross-fill.  (acceptance criteria 3 and 4)
# ---------------------------------------------------------------------------


def test_cross_fill_fills_exactly_the_missing_genes(tmp_path):
    """Criterion 3, as amended by A5: the cross-fill source is the tier-1
    ``rnaseq_basal_tpms`` key, which is the role ecoli-sources declares for it —
    NOT a column of the legacy wide table.
    """
    raw = _RawData(
        _wide_rows([("EG10001", 5.0), ("EG10002", 7.0), ("EG10003", 9.0)]),
        tables={
            RNASEQ_EXPERIMENTAL_KEY: _tpm_frame([("EG10001", 111.0)]),
            RNASEQ_BASAL_KEY: _tpm_frame(
                [("EG10001", 1.0), ("EG10002", 22.0), ("EG10003", 33.0)]),
        },
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    seq_data, prov, caught = _resolve(raw, _SimData("experimental"))

    assert seq_data["EG10001"] == 111.0            # measured: experimental wins
    assert seq_data["EG10002"] == 22.0             # unmeasured: tier 1, not 7.0
    assert seq_data["EG10003"] == 33.0             # ...and not 9.0
    assert prov["cross_fill"] == {"enabled": True, "ran": True, "n_filled": 2}

    filled = [m for m in caught if "were filled from" in m]
    assert len(filled) == 1, "cross-fill must warn ONCE, with the count"
    assert filled[0].startswith("2 gene(s)")


def test_cross_fill_off_zero_fills_and_drops_rnaseq_coverage(tmp_path):
    """Criterion 4, as amended by A9: assert the ACTUAL behaviour on both axes.

    Uncovered genes fall to ``seq_data.get(gene_id, 0.0)`` — zero — and are also
    absent from ``seq_data``, which is what sets ``_cistron_is_rnaseq_covered``
    False for them. That flag is not cosmetic: ``transcription.py`` gates the
    short-mRNA / zero-expression operon correction on it, so "cross-fill off"
    changes which cistrons get corrected, not only their values.
    """
    raw = _RawData(
        _wide_rows([("EG10001", 5.0), ("EG10002", 7.0)]),
        tables={
            RNASEQ_EXPERIMENTAL_KEY: _tpm_frame([("EG10001", 111.0)]),
            RNASEQ_BASAL_KEY: _tpm_frame([("EG10002", 22.0)]),
        },
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    seq_data, prov, _ = _resolve(raw, _SimData("experimental", rnaseq_cross_fill=False))

    assert seq_data == {"EG10001": 111.0}
    assert seq_data.get("EG10002", 0.0) == 0.0     # zero expression
    assert "EG10002" not in seq_data               # ...and not RNA-seq-covered
    assert prov["cross_fill"] == {"enabled": False, "ran": False, "n_filled": 0}


def test_cross_fill_without_a_tier_1_key_warns_rather_than_silently_zeroing(tmp_path):
    raw = _RawData(
        _wide_rows([("EG10001", 5.0), ("EG10002", 7.0)]),
        tables={RNASEQ_EXPERIMENTAL_KEY: _tpm_frame([("EG10001", 111.0)])},
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    _, prov, caught = _resolve(raw, _SimData("experimental"))

    assert prov["cross_fill"] == {"enabled": True, "ran": False, "n_filled": 0}
    assert any(RNASEQ_BASAL_KEY in m and "left at zero" in m for m in caught)


# ---------------------------------------------------------------------------
# 4. The four guards.  (acceptance criterion 9, plus A3 and A5)
# ---------------------------------------------------------------------------


def test_guard_declared_experimental_but_fitted_reference(tmp_path):
    """A3 — the stamp is the enforcement.

    The key is not in the bundle handed to this build, so the fit silently falls
    back. The stamp records ``reference`` while the config said ``experimental``,
    and the mismatch between the two is what raises the alarm — an inert selector
    cannot survive it.
    """
    raw = _RawData(_wide_rows([("EG10001", 5.0)]))
    _, prov, caught = _resolve(raw, _SimData("experimental"))

    assert prov["source"] == "reference"
    mismatch = [m for m in caught if "rnaseq source mismatch" in m]
    assert len(mismatch) == 1
    assert "'experimental'" in mismatch[0] and "'reference'" in mismatch[0]


def test_guard_variant_generated_key_is_ignored(tmp_path):
    """Criterion 9(i) — the exact failure that motivated the brief.

    A ``knockdown()`` bundle generated an expression table; the build is fitting
    the reference tier and will not read it. Loud, not silent.
    """
    expt = _write_tpm(tmp_path / "kd.tsv", [("EG10001", 0.1)])
    manifest = _variant_bundle(tmp_path, {RNASEQ_EXPERIMENTAL_KEY: expt},
                               generated=[RNASEQ_EXPERIMENTAL_KEY])
    bundle = SourceBundle(base_manifest=manifest)
    assert bundle.variant_generated_keys == {RNASEQ_EXPERIMENTAL_KEY}

    raw = _RawData(_wide_rows([("EG10001", 5.0)]), bundle=bundle)
    _, prov, caught = _resolve(raw, _SimData("reference"))

    assert prov["source"] == "reference"
    assert any("has no effect on the fit" in m for m in caught)


def test_guard_experimental_resolving_to_the_shipped_default(tmp_path):
    """Criterion 9(ii) — asked for experimental data, got the shipped table.

    True of every ``rnaseq_source: experimental`` build against the base bundle,
    because the reference bundle points the key at the basal reference by design
    (*"Defaults to the basal reference; cross-fill is a no-op when not swapped"*).
    """
    from ecoli_sources import BUNDLE_PATH

    shipped = SourceBundle(base_manifest=BUNDLE_PATH).path(RNASEQ_EXPERIMENTAL_KEY)
    raw = _RawData(
        _wide_rows([("EG10001", 5.0)]),
        tables={RNASEQ_EXPERIMENTAL_KEY: load_tpm_table(shipped)},
        sources={RNASEQ_EXPERIMENTAL_KEY: str(shipped)},
    )
    _, prov, caught = _resolve(raw, _SimData("experimental"))

    assert prov["source"] == "experimental"
    assert any("ships by default" in m for m in caught)


def test_guard_basal_expression_condition_does_not_govern_experimental(tmp_path):
    """A5's third guard — otherwise ``basal_expression_condition`` becomes the
    next silently-inert parameter: it selects a wide-table column, and the
    experimental path cross-fills from a bundle key instead.
    """
    raw = _RawData(
        _wide_rows([("EG10001", 5.0)], condition="M9 Glucose plus AAs"),
        tables={RNASEQ_EXPERIMENTAL_KEY: _tpm_frame([("EG10001", 111.0)])},
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    sim = _SimData("experimental", basal_expression_condition="M9 Glucose plus AAs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tx.resolve_basal_seq_data(raw, sim, ["EG10001"])

    assert any("does not govern the experimental rnaseq path" in str(w.message)
               for w in caught)


# ---------------------------------------------------------------------------
# 5. The provenance stamp.  (acceptance criterion 8)
# ---------------------------------------------------------------------------


def test_two_builds_are_distinguishable_from_the_stamp_alone(tmp_path, kb,
                                                             model_gene_ids):
    """Acceptance criterion 8, and it is deliberately the hard version.

    The two builds differ ONLY in the selector, and against the shipped bundle
    they are numerically identical (the shipped experimental default reproduces
    the wide table's basal column exactly). So nothing about the expression can
    tell them apart — only the stamp can. That is what "visible in our science
    and reporting" has to mean.

    This also pins the D4a shape requirement: the record must be serializable,
    carry no live objects and no absolute paths, so a later caller can copy it
    into a run's ``design`` dict verbatim.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_seq, ref = tx.resolve_basal_seq_data(
            kb, _SimData("reference"), model_gene_ids)
        exp_seq, exp = tx.resolve_basal_seq_data(
            kb, _SimData("experimental"), model_gene_ids)

    # Numerically indistinguishable...
    assert ref_seq == exp_seq
    # ...but not from the artifact.
    assert ref["source"] == "reference" and exp["source"] == "experimental"
    assert ref["canonical_key"] != exp["canonical_key"]
    assert ref["resolved_path"] != exp["resolved_path"]
    assert ref["content_sha256"] and exp["content_sha256"]
    assert ref["content_sha256"] != exp["content_sha256"]

    for record in (ref, exp):
        json.dumps(record)                              # serializable
        for value in record.values():
            assert not isinstance(value, Path)
        assert not Path(record["resolved_path"]).is_absolute()
        assert not Path(record["bundle_manifest"]).is_absolute()
        assert record["bundle_genotype_id"]             # bundle-level identity


def test_stamp_records_the_cross_fill_count(tmp_path):
    raw = _RawData(
        _wide_rows([("EG10001", 5.0), ("EG10002", 7.0)]),
        tables={
            RNASEQ_EXPERIMENTAL_KEY: _tpm_frame([("EG10001", 111.0)]),
            RNASEQ_BASAL_KEY: _tpm_frame([("EG10002", 22.0)]),
        },
        sources={RNASEQ_EXPERIMENTAL_KEY: str(tmp_path / "expt.tsv")},
    )
    _, prov, _ = _resolve(raw, _SimData("experimental"))
    assert prov["cross_fill"]["n_filled"] == 1


# ---------------------------------------------------------------------------
# 6. Plumbing — both entry points can set the selector.
# ---------------------------------------------------------------------------


def test_generator_declares_the_selector_params():
    """Declared on the REGISTRY ENTRY, or a study naming them is a hard audit
    failure (see tests/test_parca_genotype_declaration.py)."""
    from v2ecoli.composites.parca import parca

    declared = parca._composite_generator_entry.parameters
    assert declared["rnaseq_source"]["default"] == "reference"
    assert declared["rnaseq_cross_fill"]["default"] is True
    # The pre-existing params must survive.
    assert {"debug", "cpus", "cache_dir", "bundle_manifest", "bundle_overrides",
            "new_genes"} <= set(declared)


def test_generator_carries_the_selector_into_the_document():
    from v2ecoli.composites.parca import parca

    document = parca(rnaseq_source="experimental", rnaseq_cross_fill=False)["state"]
    config = document["initialize"]["config"]
    assert config["rnaseq_source"] == "experimental"
    assert config["rnaseq_cross_fill"] is False


def test_initialize_step_forwards_the_selector_to_sim_data(monkeypatch):
    """The seam that makes the CLI path work too.

    ``new_genes`` reaches the KnowledgeBase, so it only bites where the KB is
    BUILT (the composite path) — the CLI injects one and needs its own flag.
    These two reach ``sim_data.initialize`` instead, which both entry points
    call, so one field governs both.
    """
    from v2ecoli.processes.parca.steps.step_01_initialize import InitializeStep

    seen = {}

    class _FakeSimData:
        def initialize(self, **kwargs):
            seen.update(kwargs)
            raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(
        "v2ecoli.processes.parca.steps.step_01_initialize.SimulationDataEcoli",
        _FakeSimData)

    step = InitializeStep.__new__(InitializeStep)
    step.config = {"raw_data": type("_KB", (), {"operons_on": True})(),
                   "rnaseq_source": "experimental", "rnaseq_cross_fill": False}
    with pytest.raises(_Stop):
        step.update({})

    assert seen["rnaseq_source"] == "experimental"
    assert seen["rnaseq_cross_fill"] is False


def test_cli_exposes_the_selector():
    """Both entry points must be able to SET it, or the composite param is
    effective on one path only — the defect class this change removes.

    Asserts on the PARSER's behaviour, not on the source text: a grep for
    ``rnaseq_source=args.rnaseq_source`` passes just as happily when the flag
    is parsed and then dropped on the floor, which is the failure this test
    exists to catch.
    """
    import v2ecoli.cli.parca as cli_parca

    parser = cli_parca._build_arg_parser()

    # the flags exist, with the documented defaults
    defaults = parser.parse_args([])
    assert defaults.rnaseq_source == "reference"
    assert defaults.rnaseq_cross_fill is True

    # ...and they actually carry a non-default value through parsing
    flipped = parser.parse_args(["--rnaseq-source", "experimental",
                                 "--no-rnaseq-cross-fill"])
    assert flipped.rnaseq_source == "experimental"
    assert flipped.rnaseq_cross_fill is False

    # the parser rejects a value the resolver would not understand, rather
    # than passing it through to fail (or silently no-op) deep in the build
    with pytest.raises(SystemExit):
        parser.parse_args(["--rnaseq-source", "not-a-tier"])


def test_unknown_selector_value_raises():
    """Fail early and by name. A typo'd selector quietly falling back to the
    reference tier is precisely what this change exists to prevent."""
    from v2ecoli.processes.parca.reconstruction.ecoli.simulation_data import (
        SimulationDataEcoli,
    )

    sim_data = SimulationDataEcoli()
    with pytest.raises(ValueError, match="rnaseq_source must be one of"):
        sim_data.initialize(raw_data=None, rnaseq_source="expermiental")


# --- review follow-ups: the override-chain blind spot, and the real call site --


def _override_manifest(tmp_path: Path, overrides: dict) -> Path:
    """An OVERRIDE manifest — a partial table with no ``genotype.json`` sidecar.

    This is what ``--bundle-overrides`` delivers, and it is deliberately NOT a
    variant bundle: nothing writes a sidecar beside it, so a guard that reads
    only ``variant_generated_keys`` cannot see anything it supplies.
    """
    out = tmp_path / "overlay_manifest.tsv"
    pd.DataFrame(
        [{"canonical_key": k, "source_path": str(Path(v).resolve()),
          "description": "test overlay", "schema_name": ""}
         for k, v in overrides.items()]
    ).to_csv(out, sep="\t", index=False)
    return out


def test_guard_fires_for_a_key_supplied_by_the_OVERRIDE_chain(tmp_path):
    """Review finding G4 — the blind spot the base PR's own feature opens.

    An overlay supplies the experimental table through the override chain and
    the build fits the reference tier. The sidecar knows nothing about it, so a
    ``variant_generated_keys``-only guard stays silent here — reproducing the
    defect the guard exists to catch, on the newest way of supplying the key.
    """
    from ecoli_sources import BUNDLE_PATH

    expt = _write_tpm(tmp_path / "overlay.tsv", [("EG10001", 0.1)])
    overlay = _override_manifest(tmp_path, {RNASEQ_EXPERIMENTAL_KEY: expt})
    bundle = SourceBundle(base_manifest=Path(BUNDLE_PATH), overrides=overlay,
                          validate=False)

    # the sidecar path is genuinely blind to it — that is the premise
    assert bundle.variant_generated_keys == set()
    # ...and the union is not
    assert RNASEQ_EXPERIMENTAL_KEY in bundle.override_supplied_keys
    assert RNASEQ_EXPERIMENTAL_KEY in bundle.externally_supplied_keys

    raw = _RawData(_wide_rows([("EG10001", 5.0)]), bundle=bundle)
    _, prov, caught = _resolve(raw, _SimData("reference"))

    assert prov["source"] == "reference"
    assert any("has no effect on the fit" in m for m in caught), (
        "an override-supplied experimental table was ignored silently")


def test_the_real_call_site_stamps_where_it_chooses():
    """Review finding G6 — every other test calls ``resolve_basal_seq_data``
    directly, so the one production call site was covered only incidentally by
    the base PR's tests. Pin it: reorder the stack and this still holds.

    The site is ``Transcription._build_cistron_data`` (verified, not assumed —
    an earlier version of this test asserted ``__init__`` and failed, which is
    the whole reason to pin the location rather than trust a memory of it).
    """
    import inspect

    from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.process import (
        transcription as tx,
    )

    src = inspect.getsource(tx.Transcription._build_cistron_data)
    assert "resolve_basal_seq_data(" in src, (
        "the resolver is no longer called from Transcription.__init__")
    assert "sim_data.rnaseq_provenance" in src, (
        "the stamp is no longer written where the source is chosen — the "
        "stamp-is-the-record property depends on these being one step")
