"""Byte-identity tripwire for the ecoli-sources bundle.

Once the local ``flat/`` tree is deleted, the legacy-vs-bundle parity tests
(``test_kb_bundle_parity``, ``test_inherited_keys_match_ecoli_sources_content``)
skip — so nothing else actively guards the migration's core promise: that ParCa
reads the exact bytes parity was validated against.

This test pins SHA-256 of a few representative inherited keys (served from the
rev-pinned ``ecoli-sources`` package) and the 3 v2ecoli override keys (served
from local ``flat_overrides/``). It does NOT skip. If someone bumps the
``ecoli-sources`` pin in ``pyproject.toml`` to content-different data, or
changes an override file, this fails loudly — forcing a deliberate
re-validation of ParCa parity and an update of these pins.

It also asserts the source split is correct: override keys resolve into
``flat_overrides/``; inherited keys resolve out of the installed ecoli-sources
package (NOT ``flat_overrides/``).
"""

import hashlib

from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

# SHA-256 of each key's resolved file, captured against the ecoli-sources rev
# pinned in pyproject.toml at migration time. Update deliberately (and re-verify
# ParCa parity) if you bump the pin.
EXPECTED_SHA256 = {
    # Inherited from ecoli-sources:
    "genes": "a1273d3ac869e3927f02c2f29cd66cccef0cafdc656dd0a7354f1c035c50119e",
    "rnas": "bddcd413a0ebb04b912844bde21bb189047d128a7629cbd409a7166158541f53",
    "metabolites": "b87e39d72a33954f3f564db39b75810941204f1e22ebeaf077e412460157b864",
    "sequence": "28a7167d8bab60570cd6e3ddacdde75d5c51db60e1009fb8c241f1da446b6152",
    # v2ecoli local overrides (diverged biology, must win):
    "equilibrium_reactions": "836a0acb5bb347c474e7cfc63ee3599a91baa35d7bb7e9b4da45b0f09933080a",
    "equilibrium_reaction_rates": "554add4a8e94ab2bdda0d878d0998fa62974c056e9cbbd1e99154c90bea7b7f9",
    "metabolic_reactions_added": "97765aafda3c76445caedb887f96ec874a0defa7004d94e1083e89dcf68b0271",
}

OVERRIDE_KEYS = {
    "equilibrium_reactions",
    "equilibrium_reaction_rates",
    "metabolic_reactions_added",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_content_pins():
    bundle = SourceBundle()
    mismatches = []
    for key, expected in EXPECTED_SHA256.items():
        actual = _sha256(bundle.path(key))
        if actual != expected:
            mismatches.append(f"{key}: expected {expected[:12]}.., got {actual[:12]}..")
    assert mismatches == [], (
        "Bundle content changed (ecoli-sources pin bump or override edit?). "
        "Re-validate ParCa parity, then update EXPECTED_SHA256:\n  "
        + "\n  ".join(mismatches)
    )


def test_override_keys_resolve_local_and_inherited_keys_resolve_upstream():
    bundle = SourceBundle()
    for key in EXPECTED_SHA256:
        resolved = str(bundle.path(key))
        if key in OVERRIDE_KEYS:
            assert "flat_overrides" in resolved, f"{key} should be a local override"
        else:
            assert "flat_overrides" not in resolved, (
                f"{key} should come from ecoli-sources, not a local override"
            )
