"""Regression test for the aa_in_media pint-Quantity-vs-float comparison in
``SteadyStatePolypeptideElongation._amino_acid_supply``.

On AA-containing media (with_aa / succinate) ``states["boundary"]["external"]``
values arrive as pint Quantities rather than bare floats, while
``import_constraint_threshold`` is a plain float in mM. Comparing a Quantity
directly to a float raises ``ValueError: Cannot compare PlainQuantity and
float``. This used to be carried only as a downstream monkeypatch in
sms-ecoli (``pbg_v2ecoli/_upstream_patches.py``); it is now fixed directly in
``_amino_acid_supply``.

The probe below invokes the real (unbound) method with a deliberately
incomplete stub ``self`` -- just enough to exercise the ``aa_in_media``
comprehension -- and reads how far execution gets, mirroring the detection
probe in sms-ecoli's monkeypatch module.
"""
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.fast


def _invoke_amino_acid_supply(fake_self, fake_states):
    from v2ecoli.processes.polypeptide_elongation import (
        SteadyStatePolypeptideElongation,
    )

    method = SteadyStatePolypeptideElongation._amino_acid_supply
    return method(fake_self, fake_states, aa_conc=None, dry_mass=None, counts_to_uM_mag=None)


def test_aa_in_media_accepts_pint_quantity_external_concentration():
    """A pint-Quantity boundary-external value must not raise when compared
    against the bare-float import_constraint_threshold."""
    from bigraph_schema.units import units as ureg

    fake_self = SimpleNamespace(
        aa_environment_names=["LEU"],
        import_constraint_threshold=0.1,
    )
    fake_states = {"boundary": {"external": {"LEU": 1.0 * ureg.mM}}}

    with pytest.raises(AttributeError) as excinfo:
        # The stub self is missing everything past the aa_in_media
        # comparison (get_pathway_enzyme_counts_per_aa first), so clearing
        # the comparison surfaces as an AttributeError on that next
        # attribute access -- not the ValueError the pre-fix code raised.
        _invoke_amino_acid_supply(fake_self, fake_states)
    assert "get_pathway_enzyme_counts_per_aa" in str(excinfo.value)


def test_aa_in_media_still_compares_bare_floats():
    """Bare-float external concentrations (no pint units) keep working."""
    fake_self = SimpleNamespace(
        aa_environment_names=["LEU"],
        import_constraint_threshold=0.1,
    )
    fake_states = {"boundary": {"external": {"LEU": 1.0}}}

    with pytest.raises(AttributeError) as excinfo:
        _invoke_amino_acid_supply(fake_self, fake_states)
    assert "get_pathway_enzyme_counts_per_aa" in str(excinfo.value)
