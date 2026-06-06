import os
from pathlib import Path

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.sources import (
    SourceBundle,
    relpath_to_key,
)


def test_relpath_to_key_strips_ext_and_joins_with_double_underscore():
    assert relpath_to_key("genes.tsv") == "genes"
    assert relpath_to_key(os.path.join("condition", "media", "MIX0-55.tsv")) == "condition__media__MIX0-55"
    assert relpath_to_key(os.path.join("cell_wall", "murein_strand_length_distribution.csv")) == "cell_wall__murein_strand_length_distribution"


def test_default_bundle_resolves_a_known_key():
    b = SourceBundle()
    p = b.path("genes")
    assert p.is_file()


def test_resolve_relpath_routes_through_key():
    b = SourceBundle()
    assert b.resolve_relpath("genes.tsv") == b.path("genes")


def test_missing_key_raises_naming_the_key():
    b = SourceBundle()
    with pytest.raises(KeyError, match="no_such_key"):
        b.path("no_such_key")


def test_override_replaces_base_row(tmp_path):
    # base manifest with one key
    data_root = tmp_path / "data"
    (data_root / "flat").mkdir(parents=True)
    (data_root / "flat" / "genes.tsv").write_text("base")
    base = data_root / "reference_bundle.tsv"
    base.write_text("canonical_key\tsource_path\tdescription\tschema_name\n"
                    "genes\tflat/genes.tsv\tg\t\n")
    # override pointing genes elsewhere
    ov_root = tmp_path / "ov"
    (ov_root / "flat_overrides").mkdir(parents=True)
    (ov_root / "flat_overrides" / "genes.tsv").write_text("override")
    ov = ov_root / "parca_overrides.tsv"
    ov.write_text("canonical_key\tsource_path\tdescription\tschema_name\n"
                  "genes\tflat_overrides/genes.tsv\tg\t\n")

    b = SourceBundle(base_manifest=base, overrides=ov, validate=False)
    assert b.path("genes").read_text() == "override"
