"""
The `parca` composite generator must let a build name the GENOTYPE it is for.

A ParCa build's genotype is the ecoli-sources bundle manifest its raw_data was
built from -- a knockout genome is a different manifest. Before this, the
generator accepted only debug/cpus/cache_dir, so a knockout build and a
wild-type build were indistinguishable in composite params and a study had no
schema-validated place to say which genome it built.

These are declarative parameters (the KB is still constructed and injected by
the runners), so the tests assert the identity is CARRIED and CHECKED, not that
it builds a knowledge base.
"""

import warnings

import pytest

from v2ecoli.composites.parca import parca
from v2ecoli.processes.parca.composite import build_parca_document
from v2ecoli.processes.parca.steps.step_01_initialize import InitializeStep

MANIFEST = "/tmp/genotype-ko-EG10526/bundle.tsv"
OVERRIDES = "/tmp/genotype-ko-EG10526/overrides.tsv"


def _initialize_config(document):
    return document["initialize"]["config"]


# --------------------------------------------------------------------------
# The generator accepts and carries the identity.
# --------------------------------------------------------------------------


def test_generator_accepts_bundle_parameters():
    """
    The study-surface unblock: these must be declared on the REGISTRY ENTRY,
    since that is what the audit's params-are-generator-accepted check reads.
    A study naming a param absent from here is a hard audit failure -- which is
    why showcase-1-parca's `mode` is allowlisted.
    """
    declared = parca._composite_generator_entry.parameters

    assert "bundle_manifest" in declared
    assert "bundle_overrides" in declared
    assert declared["bundle_manifest"]["default"] == ""
    # The pre-existing three must survive.
    assert {"debug", "cpus", "cache_dir"} <= set(declared)


def test_generator_carries_the_genotype_into_the_document():
    document = parca(bundle_manifest=MANIFEST, bundle_overrides=OVERRIDES)["state"]

    config = _initialize_config(document)
    assert config["bundle_manifest"] == MANIFEST
    assert config["bundle_overrides"] == OVERRIDES


def test_two_genotypes_produce_distinguishable_documents():
    """
    The property that was missing: a knockout build and a wild-type build must
    not be identical documents.
    """
    wild_type = parca()["state"]
    knockout = parca(bundle_manifest=MANIFEST)["state"]

    assert _initialize_config(wild_type) != _initialize_config(knockout)
    assert _initialize_config(wild_type)["bundle_manifest"] == ""


def test_document_builder_defaults_to_wild_type():
    """Omitting the genotype must stay valid -- it means the default bundle."""
    config = _initialize_config(build_parca_document())

    assert config["bundle_manifest"] == ""
    assert config["bundle_overrides"] == ""
    assert config["raw_data"] is None


def test_step_declares_the_config_fields():
    assert InitializeStep.config_schema["bundle_manifest"]["_default"] == ""
    assert InitializeStep.config_schema["bundle_overrides"]["_default"] == ""


# --------------------------------------------------------------------------
# The cross-check: a declared genotype that disagrees with the injected
# raw_data must not fit silently.
# --------------------------------------------------------------------------


class _FakeBundle:
    def __init__(self, base_manifest):
        self.base_manifest = base_manifest


class _FakeRawData:
    def __init__(self, base_manifest):
        self._bundle = _FakeBundle(base_manifest)


def _check(declared, raw_data):
    step = object.__new__(InitializeStep)
    step.config = {"bundle_manifest": declared, "raw_data": raw_data}
    step._check_declared_genotype()


def test_mismatched_genotype_warns():
    """
    The expensive failure this guards: the fit succeeds and its sim_data is
    attributed to a genotype it was not built from.
    """
    with pytest.warns(UserWarning, match="genotype mismatch"):
        _check(MANIFEST, _FakeRawData("/tmp/some-other-genotype/bundle.tsv"))


def test_matching_genotype_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(MANIFEST, _FakeRawData(MANIFEST))


def test_equivalent_paths_do_not_warn():
    """Path spelling is not a genotype difference."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check("/tmp/genotype-ko-EG10526/../genotype-ko-EG10526/bundle.tsv",
               _FakeRawData(MANIFEST))


@pytest.mark.parametrize(
    "declared,raw_data",
    [
        ("", _FakeRawData(MANIFEST)),          # nothing declared -> nothing to check
        (MANIFEST, None),                       # no raw_data injected yet
        (MANIFEST, object()),                   # raw_data without a bundle
    ],
)
def test_check_is_silent_when_it_cannot_conclude(declared, raw_data):
    """
    The check must never warn on absence of information -- a structural document
    legitimately carries raw_data=None.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(declared, raw_data)
