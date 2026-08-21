"""Every medium a recipe references must be one the KB loader actually reads.

``LIST_OF_DICT_FILENAMES`` in ``knowledge_base_raw`` hardcodes which media
ingredient files are loaded. Nothing ties that hand-maintained list to
``media_recipes.tsv``, which is what names the media a build needs. A recipe
can therefore reference a medium that is shipped, declared in the bundle
manifest and resolvable — and still never read, because no entry in the list
asks for it. The build does not fail; the medium is simply absent.

The check takes a bundle rather than looking one up, because the case that
bites arrives through the override chain: a recipe contributed by
``--bundle-overrides`` can name a medium the public reference bundle never
had. A test pinned to the reference bundle would be green in exactly the
situation it exists to catch — so the parameter is the point of the test, not
a convenience.
"""

import os
import shutil

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    LIST_OF_DICT_FILENAMES,
    media_registration_gaps,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle


def test_reference_bundle_registers_every_medium_its_recipes_reference():
    """The invariant, on the bundle every stock build uses."""
    assert media_registration_gaps(SourceBundle()) == {}


def test_gap_is_detected_when_a_recipe_names_an_unregistered_medium(tmp_path):
    """The same check, against a bundle that violates the invariant.

    Without this, the test above is indistinguishable from one that computes
    an empty set for the wrong reason. The violating medium is a real shipped
    file that no recipe currently references and that the loader does not
    list, so the fixture is the actual defect shape rather than an invented
    one.
    """
    bundle = SourceBundle()
    unregistered = _a_shipped_but_unregistered_medium(bundle)

    recipes = tmp_path / "media_recipes.tsv"
    src = bundle.path("condition__media_recipes")
    shutil.copy(src, recipes)
    with open(recipes, "a") as f:
        f.write(
            '\t'.join([
                '"minimal_plus_test_probe"', f'"{unregistered}"', '1.0',
                '""', '0', '[]', '[]', '[]', '[]',
            ]) + "\n"
        )

    manifest = tmp_path / "overrides.tsv"
    manifest.write_text(
        "canonical_key\tsource_path\tdescription\tschema_name\n"
        "condition__media_recipes\tmedia_recipes.tsv\tprobe\t\n"
    )
    probed = SourceBundle(overrides=[str(manifest)])

    gaps = media_registration_gaps(probed)
    assert unregistered in gaps, (
        f"a recipe referencing the unloaded medium {unregistered!r} was not "
        f"reported; the check cannot see the defect it exists to catch"
    )
    assert gaps[unregistered] == "not in LIST_OF_DICT_FILENAMES"


def _a_shipped_but_unregistered_medium(bundle):
    """A medium present in the bundle but absent from the loader's list."""
    registered = {
        os.path.splitext(os.path.basename(f))[0]
        for f in LIST_OF_DICT_FILENAMES
        if os.path.dirname(f) == os.path.join("condition", "media")
    }
    shipped = {
        key.rsplit("__", 1)[-1]
        for key in bundle.keys_with_prefix("condition__media__")
    }
    candidates = sorted(shipped - registered)
    if not candidates:
        pytest.skip(
            "no shipped-but-unregistered medium available to build the "
            "negative case from"
        )
    return candidates[0]
