import hashlib
import json
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


def _variant_bundle(tmp_path, generated_keys):
    """Build a base + a variant bundle (manifest + genotype.json sidecar) + a
    v2ecoli-style override that collides on ``dna_sites``.

    Mirrors ecoli-sources ``compose_variant_bundle``: the variant manifest is a
    full manifest where the generated keys point at the variant's OWN files and
    every other key is inherited from base, and a ``genotype.json`` sidecar
    records ``overridden_keys`` — the authoritative list of what the variant
    generated. Returns the constructed SourceBundle.
    """
    base_root = tmp_path / "base"
    (base_root / "flat").mkdir(parents=True)
    (base_root / "flat" / "dna_sites.tsv").write_text("base-dna_sites")
    (base_root / "flat" / "equilibrium_reactions.tsv").write_text("base-eq")
    base = base_root / "reference_bundle.tsv"
    base.write_text(
        "canonical_key\tsource_path\tdescription\tschema_name\n"
        "dna_sites\tflat/dna_sites.tsv\td\t\n"
        "equilibrium_reactions\tflat/equilibrium_reactions.tsv\te\t\n"
    )

    # The variant out_dir: a knockout regenerates dna_sites into its own dir and
    # inherits equilibrium_reactions by pointing back at base.
    vdir = tmp_path / "variant"
    vdir.mkdir()
    (vdir / "dna_sites.tsv").write_text("variant-dna_sites")
    vman = vdir / "reference_bundle.tsv"
    eq_rel = os.path.relpath(base_root / "flat" / "equilibrium_reactions.tsv", start=vdir)
    vman.write_text(
        "canonical_key\tsource_path\tdescription\tschema_name\n"
        f"dna_sites\tdna_sites.tsv\td\t\n"
        f"equilibrium_reactions\t{eq_rel}\te\t\n"
    )
    (vdir / "genotype.json").write_text(json.dumps({"overridden_keys": sorted(generated_keys)}))

    # v2ecoli override: whole-file overrides that collide on BOTH keys.
    ov_root = tmp_path / "ov"
    (ov_root / "flat_overrides").mkdir(parents=True)
    (ov_root / "flat_overrides" / "dna_sites.tsv").write_text("override-dna_sites")
    (ov_root / "flat_overrides" / "equilibrium_reactions.tsv").write_text("override-eq")
    ov = ov_root / "parca_overrides.tsv"
    ov.write_text(
        "canonical_key\tsource_path\tdescription\tschema_name\n"
        "dna_sites\tflat_overrides/dna_sites.tsv\td\t\n"
        "equilibrium_reactions\tflat_overrides/equilibrium_reactions.tsv\te\t\n"
    )
    return SourceBundle(base_manifest=vman, overrides=ov, validate=False)


def test_variant_generated_key_wins_over_override(tmp_path):
    # #466: a knockout regenerates dna_sites; the whole-file override must NOT
    # clobber it back to pre-deletion coordinates.
    b = _variant_bundle(tmp_path, generated_keys={"dna_sites"})
    assert b.path("dna_sites").read_text() == "variant-dna_sites"


def test_override_still_wins_for_keys_the_variant_did_not_generate(tmp_path):
    # The override's purpose — v2's diverged equilibrium/metabolism files win
    # over the shared base — must survive the fix for keys the variant inherits.
    b = _variant_bundle(tmp_path, generated_keys={"dna_sites"})
    assert b.path("equilibrium_reactions").read_text() == "override-eq"


V2_FLAT = Path(__file__).resolve().parents[1] / "v2ecoli/processes/parca/reconstruction/ecoli/flat"
OVERRIDE_KEYS = {"equilibrium_reactions", "equilibrium_reaction_rates", "metabolic_reactions_added"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_override_keys_point_to_local_flat_overrides():
    b = SourceBundle()
    for key in OVERRIDE_KEYS:
        p = b.path(key)
        assert "flat_overrides" in str(p), f"{key} should resolve to a local override"
        assert p.is_file()


@pytest.mark.skipif(not V2_FLAT.exists(), reason="local flat/ deleted post-migration")
def test_inherited_keys_match_ecoli_sources_content():
    # For every still-present local flat file NOT overridden, the bundle's
    # resolved file must be byte-identical (guards against upstream drift).
    b = SourceBundle()
    mismatches = []
    for f in V2_FLAT.rglob("*"):
        if not f.is_file() or f.name == "sequence.fasta":
            continue
        rel = f.relative_to(V2_FLAT)
        key = relpath_to_key(str(rel))
        if key in OVERRIDE_KEYS or not b.has_key(key):
            continue
        if _sha(f) != _sha(b.path(key)):
            mismatches.append(key)
    assert mismatches == [], f"bundle content drifted from local flat for: {mismatches}"
