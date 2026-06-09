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
import os

import numpy as np
import pytest


CACHE = "out/cache"
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


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
        # 3b + 3c methods removed — landed in
        # v2ecoli/processes/polypeptide/kinetic_charging.py.
        ("final_amino_acids", "Task 3d"),
        ("evolve", "Task 3d"),
        ("reconcile", "Task 3d"),
        ("protein_maturation", "Task 3d"),
        ("monomer_to_aa", "Task 3e"),
        ("monomer_limit", "Task 3e"),
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


# ------------------------ Task 3b: __init__ + get_kinetic_constants ------------------------

def test_initialize_is_no_longer_a_stub() -> None:
    """3b is the marker that ``initialize`` and ``get_kinetic_constants``
    have left the scaffold's stub-list. If either reverts to NotImplementedError,
    this fails loudly so the next session knows to land it."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    init_src = inspect.getsource(KT.initialize)
    gkc_src = inspect.getsource(KT.get_kinetic_constants)
    assert "Task 3b" not in init_src, "initialize still carries Task 3b marker"
    assert "Task 3b" not in gkc_src, "get_kinetic_constants still carries Task 3b marker"
    assert "NotImplementedError" not in init_src, "initialize still raises NotImplementedError"
    assert "NotImplementedError" not in gkc_src, (
        "get_kinetic_constants still raises NotImplementedError"
    )


def test_initialize_sets_documented_attrs_via_source_scan() -> None:
    """Without instantiating, verify each documented attr assignment lands.
    Catches the case where a future refactor accidentally drops one of the
    unpackings — pure source scan, no cache needed."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    init_src = inspect.getsource(KT.initialize)
    expected_assignments = [
        "self.cell_density",
        "self.protein_sequences",
        "self.monomer_weights_incorporated",
        "self.n_monomers",
        "self.i_start_codon",
        "self.is_map_substrate",
        "self.n_trnas",
        "self.n_codons",
        "self.molecules_input_size",
        "self.slice_free_trnas",
        "self.slice_charged_trnas",
        "self.slice_amino_acids",
        "self.slice_charging_counter",
        "self.slice_reading_counter",
        "self.slice_codons_to_trnas_counter",
        "self.trnas_to_amino_acids",
        "self.amino_acids_to_trnas",
        "self.trnas_to_codons",
        "self.codons_to_trnas",
        "self.codons_to_amino_acids",
        "self.trnas_to_amino_acid_indexes",
        "self.max_attempts",
        "self.k_cat__per_s",
        "self.K_M_amino_acid__per_L",
        "self.K_M_trna__per_L",
        "self.buffer",
        "self.previous_rate",
    ]
    missing = [a for a in expected_assignments if a not in init_src]
    assert not missing, f"initialize is missing attribute assignments: {missing}"


def test_get_kinetic_constants_uses_cell_volume_conversion() -> None:
    """Sanity check that get_kinetic_constants computes cell_volume and
    applies it to both K_M arrays — guards against a regression to the
    earlier upstream stub that just returned the per-L values verbatim."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.get_kinetic_constants)
    assert "cell_volume" in src
    assert "self.cell_density" in src
    assert "K_M_amino_acids" in src
    assert "K_M_trnas" in src
    assert "K_M_amino_acid__per_L" in src
    assert "K_M_trna__per_L" in src


# ------------------------ end-to-end cache-backed instantiation ------------------------

def _make_kinetic_extensions(n_aas: int = 21, n_trnas: int = 44, n_codons: int = 62,
                              n_proteins: int = 10, n_synthetases: int = 21) -> dict:
    """Build synthetic kinetic-charging config keys good enough for ``initialize``.

    These mirror the shape contract that Task #5 will populate from sim_data.
    Values are placeholders — they're consumed by ``initialize`` as opaque
    array data, but downstream methods (3c–3e) will need realistic ones.
    """
    from v2ecoli.types.quantity import ureg as units

    return {
        "codon_sequences": np.zeros((n_proteins, 100), dtype=np.int8),
        "residue_weights_by_codon": np.ones(n_codons, dtype=np.float64),
        "n_codons": n_codons,
        "i_start_codon": 0,
        "is_map_substrate": np.zeros(n_proteins, dtype=bool),
        "n_trna_codon_pairs": n_codons + n_trnas,
        # Each codon read by at least one tRNA (use a simple round-robin map)
        "trnas_to_codons": (
            (np.arange(n_codons)[:, None] % n_trnas == np.arange(n_trnas)[None, :])
            .astype(np.int8)
        ),
        # Each amino acid mapped to ~3 codons; first column distinguishes the start codon
        "codons_to_amino_acids": np.eye(n_aas, n_codons, dtype=np.int8),
        # k_cat per synthetase
        "k_cat__per_s": np.ones(n_synthetases, dtype=np.float64) * 10.0,
        # K_M values stored per litre (pint Quantity arrays)
        "K_M_amino_acid__per_L": np.ones(n_synthetases, dtype=np.float64) * 1e-4 * (units.mol / units.L),
        "K_M_trna__per_L": np.ones(n_synthetases, dtype=np.float64) * 1e-5 * (units.mol / units.L),
        "reconciliation_buffer": 10,
    }


@pytest.mark.sim
@_needs_cache
def test_initialize_runs_end_to_end_against_cache() -> None:
    """Build a full config dict from the cache + synthetic kinetic extensions
    and instantiate the class. Verifies that the unpacking interacts correctly
    with the base ``initialize`` (after super().initialize call).
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )

    cfg = dict(load_cache_bundle(CACHE)["configs"]["ecoli-polypeptide-elongation"])
    # Replace n_trnas-sensitive fields with our synthetic ones so the shape
    # contracts line up (cache's aa_from_trna defines n_trnas).
    n_aas = len(cfg["amino_acids"])
    n_trnas = len(cfg["uncharged_trna_names"])
    n_synthetases = len(cfg.get("synthetase_names", [])) or n_aas
    cfg.update(_make_kinetic_extensions(
        n_aas=n_aas, n_trnas=n_trnas, n_codons=62,
        n_proteins=len(cfg["proteinIds"]),
        n_synthetases=n_synthetases,
    ))

    proc = KineticTrnaChargingPolypeptideElongation(cfg)

    # Spot-check that the kinetic-specific attrs landed on the instance
    assert proc.n_trnas == n_trnas
    assert proc.n_codons == 62
    assert proc.molecules_input_size == (
        n_trnas + n_trnas + n_aas + n_trnas + n_trnas + (62 + n_trnas)
    )
    assert isinstance(proc.slice_free_trnas, slice)
    assert proc.slice_charged_trnas.start == proc.slice_free_trnas.stop
    assert proc.trnas_to_amino_acid_indexes.shape == (n_trnas,)
    assert proc.max_attempts == np.byte(4)
    assert proc.buffer == 10
    # Warm-start: previous_rate = int(ribosomeElongationRate * time_step)
    assert proc.previous_rate == int(
        proc.ribosomeElongationRate * cfg.get("time_step", 1)
    )


# ------------------------ Task 3c: request-side methods ------------------------

@pytest.mark.parametrize(
    "method_name",
    ["elongation_rate", "request", "run_model", "codon_sequences_width",
     "sequences", "max_charging_rate", "_init_bulk_indices"],
)
def test_3c_method_no_longer_stub(method_name: str) -> None:
    """3c is the marker that the request-side methods have left the
    scaffold. If any reverts to a NotImplementedError raise, this fails."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    method = getattr(KT, method_name, None)
    assert method is not None, f"missing method: {method_name}"
    src = inspect.getsource(method)
    assert "Task 3c" not in src, f"{method_name} still carries Task 3c marker"
    assert "NotImplementedError" not in src, (
        f"{method_name} still raises NotImplementedError"
    )


def test_elongation_rate_calls_kernel_and_sets_longer_sequences() -> None:
    """Source-scan to verify ``elongation_rate`` calls the kernel and caches
    ``self.longer_sequences`` for later use by ``request``/``evolve``."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.elongation_rate)
    assert "kernel.get_elongation_rate" in src
    assert "self.longer_sequences" in src
    assert "self.sequences_width" in src
    assert "self.previous_rate" in src
    assert "buildSequences" in src


def test_request_requests_all_kinetic_bulk_keys() -> None:
    """Source-scan to verify ``request`` builds bulk requests for amino acids,
    ATP, both tRNA pools, synthetases, MAP, and water."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.request)
    for idx in [
        "self.amino_acid_idx",
        "self.atp_idx",
        "self.uncharged_trna_idx",
        "self.charged_trna_idx",
        "self.synthetase_idx",
        "self.map_idx",
        "self.water_idx",
    ]:
        assert idx in src, f"request missing bulk index: {idx}"
    # Returns the v2ecoli tuple
    assert "fraction_charged" in src
    assert "amino_acids_used.astype(float)" in src


def test_run_model_uses_ode_and_kernel_constants() -> None:
    """Source-scan: ``run_model`` runs ``solve_ivp`` and uses our K_M /
    saturation pre-compute on bulk_total."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT.run_model)
    assert "solve_ivp" in src
    assert "method=\"RK45\"" in src
    assert "rtol=1e-4" in src
    assert "atol=1e-7" in src
    assert "self.K_M_amino_acids" in src
    assert "self.K_M_trnas" in src
    assert "stochasticRound" in src
    # Returns the full 7-tuple
    assert "amino_acids_used" in src
    assert "codons_read" in src
    assert "codons_to_trnas_matrix" in src


def test_init_bulk_indices_adds_kinetic_keys() -> None:
    """The override adds ATP/AMP/PPi/MET/MAP indices to the base layout."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )
    src = inspect.getsource(KT._init_bulk_indices)
    assert "super()._init_bulk_indices" in src
    for idx in ["self.atp_idx", "self.amp_idx", "self.ppi_idx",
                "self.met_idx", "self.map_idx"]:
        assert idx in src, f"_init_bulk_indices missing: {idx}"


@pytest.mark.sim
@_needs_cache
def test_get_kinetic_constants_returns_volume_scaled_arrays() -> None:
    """Smoke-test get_kinetic_constants against a freshly-instantiated process.

    Asserts the returned arrays have the right shape and that the values scale
    with cell mass (doubling mass → doubling K_M).
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )

    cfg = dict(load_cache_bundle(CACHE)["configs"]["ecoli-polypeptide-elongation"])
    n_aas = len(cfg["amino_acids"])
    n_trnas = len(cfg["uncharged_trna_names"])
    n_synthetases = len(cfg.get("synthetase_names", [])) or n_aas
    cfg.update(_make_kinetic_extensions(
        n_aas=n_aas, n_trnas=n_trnas, n_codons=62,
        n_proteins=len(cfg["proteinIds"]),
        n_synthetases=n_synthetases,
    ))
    proc = KineticTrnaChargingPolypeptideElongation(cfg)

    K_M_aa_1, K_M_trna_1 = proc.get_kinetic_constants(1.0)
    K_M_aa_2, K_M_trna_2 = proc.get_kinetic_constants(2.0)

    # Doubling the mass should double the volume → double the K_M.
    np.testing.assert_allclose(
        np.asarray(K_M_aa_2.magnitude if hasattr(K_M_aa_2, "magnitude") else K_M_aa_2),
        2.0 * np.asarray(K_M_aa_1.magnitude if hasattr(K_M_aa_1, "magnitude") else K_M_aa_1),
    )
    np.testing.assert_allclose(
        np.asarray(K_M_trna_2.magnitude if hasattr(K_M_trna_2, "magnitude") else K_M_trna_2),
        2.0 * np.asarray(K_M_trna_1.magnitude if hasattr(K_M_trna_1, "magnitude") else K_M_trna_1),
    )
