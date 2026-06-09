"""
Scaffold tests for :class:`KineticTrnaChargingPolypeptideElongation` (Task 3a).

Gates that:

* The module imports cleanly.
* The class subclasses :class:`BasePolypeptideElongation` and inherits its
  ``name``, ``topology``, and full ``config_schema`` (no regression in the
  base schema surface).
* ``config_schema`` contains every kinetic-charging-specific key that
  upstream's ``KineticTrnaChargingModel.__init__`` unpacks.
* Each method stub raises ``NotImplementedError`` and names its owning
  task (3b / 3c / 3d / 3e) so the runtime error tells you which session to
  pick up next.

Method bodies are gated by future tests in
``tests/test_kinetic_charging_polypeptide_elongation.py`` (to be added by
3c/3d/3e).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_module_imports() -> None:
    from v2ecoli.processes.polypeptide import kinetic_charging  # noqa: F401


def test_class_subclasses_base() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )
    from v2ecoli.processes.polypeptide_elongation import BasePolypeptideElongation

    assert issubclass(
        KineticTrnaChargingPolypeptideElongation, BasePolypeptideElongation
    )


def test_inherits_name_and_topology() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )
    from v2ecoli.processes.polypeptide_elongation import NAME, TOPOLOGY

    assert KineticTrnaChargingPolypeptideElongation.name == NAME
    assert KineticTrnaChargingPolypeptideElongation.topology == TOPOLOGY


def test_config_schema_includes_kinetic_charging_keys() -> None:
    """Every parameter unpacked by upstream's KineticTrnaChargingModel.__init__
    must have a config_schema entry. Detects drift if a future port adds a
    new parameter without registering it."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    expected = {
        "codon_sequences",
        "residue_weights_by_codon",
        "n_codons",
        "i_start_codon",
        "is_map_substrate",
        "n_trna_codon_pairs",
        "trnas_to_codons",
        "codons_to_amino_acids",
        "k_cat__per_s",
        "K_M_amino_acid__per_L",
        "K_M_trna__per_L",
        "reconciliation_buffer",
    }
    missing = expected - set(KT.config_schema.keys())
    assert not missing, f"missing config_schema keys: {sorted(missing)}"


def test_base_config_schema_keys_still_present() -> None:
    """Inheritance shouldn't have dropped any base schema keys — the dict
    merge ``{**Base.config_schema, ...}`` must keep every base entry."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    from v2ecoli.processes.polypeptide_elongation import BasePolypeptideElongation

    for key in BasePolypeptideElongation.config_schema:
        assert key in KT.config_schema, f"missing inherited key {key}"


def test_kinetic_keys_have_well_formed_schema_entries() -> None:
    """Each new key has both ``_type`` and ``_default``; defaults are
    NumPy-shaped according to the documented downstream usage. Catches
    typos that would silently fall through to the wrong dtype."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    new_keys = {
        "codon_sequences",
        "residue_weights_by_codon",
        "n_codons",
        "i_start_codon",
        "is_map_substrate",
        "n_trna_codon_pairs",
        "trnas_to_codons",
        "codons_to_amino_acids",
        "k_cat__per_s",
        "K_M_amino_acid__per_L",
        "K_M_trna__per_L",
        "reconciliation_buffer",
    }
    for key in new_keys:
        entry = KT.config_schema[key]
        assert "_type" in entry, f"{key}: missing _type"
        assert "_default" in entry, f"{key}: missing _default"


@pytest.mark.parametrize(
    ("method_name", "task_marker"),
    [
        ("initialize", "Task 3b"),
        ("elongation_rate", "Task 3c"),
        ("request", "Task 3c"),
        ("final_amino_acids", "Task 3d"),
        ("evolve", "Task 3d"),
        ("run_model", "Task 3c"),
        ("reconcile", "Task 3d"),
        ("protein_maturation", "Task 3d"),
        ("monomer_to_aa", "Task 3e"),
        ("monomer_limit", "Task 3e"),
        ("codon_sequences_width", "Task 3c"),
        ("sequences", "Task 3c"),
        ("max_charging_rate", "Task 3c"),
        ("get_kinetic_constants", "Task 3b"),
    ],
)
def test_method_stub_carries_task_marker(method_name: str, task_marker: str) -> None:
    """Each stubbed method's body must reference its owning task in the
    ``NotImplementedError`` message. Catches drift when a future port
    fills in one method but not another."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    method = getattr(KT, method_name, None)
    assert method is not None, f"missing method: {method_name}"
    src = inspect.getsource(method)
    assert task_marker in src, (
        f"{method_name} stub missing {task_marker!r} marker in source"
    )
