"""The genuine ``vecoli`` build translates the fixture↔upstream skew into an
actionable error.

When the sim_data fixture predates upstream vEcoli's ``modified_proteins``
feature (2026-05-01) but the vEcoli checkout is newer, upstream's
``get_monomer_counts_listener_config`` reads
``TwoComponentSystem.modified_molecules`` off a TCS that lacks it and the
composer dies deep in ``sim_data.py`` with a bare ``AttributeError`` that names
neither the cause nor the fix. ``build_vivarium_ecoli`` catches exactly that and
re-raises a message naming both. These tests cover the detection + message
helpers (unit-testable without a full ~300MB EcoliSim build).
"""
from v2ecoli.library.vivarium_ecoli_engine import (
    _is_tcs_modified_molecules_skew,
    _fixture_skew_message,
)


def test_detects_the_exact_skew():
    # The real bare error the composer raises for this skew.
    err = AttributeError(
        "'TwoComponentSystem' object has no attribute 'modified_molecules'")
    assert _is_tcs_modified_molecules_skew(err) is True


def test_ignores_unrelated_attribute_errors():
    # Both tokens are required, so an unrelated AttributeError is NOT captured
    # and keeps propagating untouched.
    assert _is_tcs_modified_molecules_skew(
        AttributeError("'Foo' object has no attribute 'bar'")) is False
    assert _is_tcs_modified_molecules_skew(
        AttributeError("'TwoComponentSystem' object has no attribute 'stoich'")) is False
    assert _is_tcs_modified_molecules_skew(
        AttributeError("'Metabolism' object has no attribute 'modified_molecules'")) is False


def test_message_names_cause_and_both_fixes():
    msg = _fixture_skew_message("/ws/out/cache/simData.cPickle", "/repos/vEcoli")
    # cause
    assert "modified_proteins" in msg and "2026-05-01" in msg
    assert "modified_molecules" in msg
    # the two remedies
    assert "7bf03433" in msg                      # (a) pin the checkout
    assert "build_upstream_parca.py" in msg       # (b) rebuild the fixture
    # names the actual paths involved
    assert "/ws/out/cache/simData.cPickle" in msg
    assert "/repos/vEcoli" in msg


def test_message_falls_back_to_env_placeholder(monkeypatch):
    # With no fork_dir and no V2E_VECOLI_DIR set, the message still renders with
    # a placeholder rather than crashing.
    monkeypatch.delenv("V2E_VECOLI_DIR", raising=False)
    msg = _fixture_skew_message("/x/simData.cPickle", None)
    assert "<V2E_VECOLI_DIR>" in msg
