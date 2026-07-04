"""Self-contained plasmid (pBR322 / ColE1) data layer — no ParCa / sim_data.

The plasmid is a small, fully-specified object: a ~4.4 kb known sequence
(pBR322, GenBank J01749.1) plus literature copy-number kinetics (Brendel &
Perelson 1993, *J Mol Biol* 229:860-872). It does **not** need the whole-cell
ParCa fit. This module builds everything ``PlasmidReplication`` needs directly
from the committed FASTA and from the *shared* replication parameters that are
already present in the baseline cache's ``ecoli-chromosome-replication`` config
(nucleotide weights, replisome mass, elongation-rate machinery, dNTP / PPi ids,
submass indices, D-period, ...).

Nothing here imports ParCa, ``sim_data``, or reads ``flat_overrides/`` — the
plasmid layer is a purely additive decoration on top of a finished baseline
document. See ``v2ecoli/composites/plasmids.py``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from v2ecoli.library.polymerize import polymerize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# pBR322 replication origin (oriV) center coordinate. The ParCa-coupled branch
# read this from ``flat_overrides/plasmid_dna_sites.tsv`` (site "P-ori",
# left_end_pos=2534, right_end_pos=3122) via ``get_site_center_coordinates`` =
# round((left + right) / 2) = round((2534 + 3122) / 2) = 2828. We bake it in as
# a documented constant so the decoupled layer needs no ParCa DNA-sites table.
# It only sets the rotation origin of the replication sequence (where the single
# unidirectional fork starts), so an exact value is not load-bearing for the
# copy-number dynamics — but we keep the literature-correct value for fidelity.
ORIV_COORDINATE = 2828

# ATGC → int8 base pairing is (n_nt_types - 1) - i; see nucleotide-index note in
# ``_build_plasmid_replication_sequences``.
_N_NT_TYPES = 4

# Matches wholecell replication.MAX_TIMESTEP — buffer (in max-timestep units)
# appended to the polymerization matrix so buildSequences never runs off the end
# within a single step.
MAX_TIMESTEP = 2

_FASTA_PATH = os.path.join(os.path.dirname(__file__), "plasmid_pbr322.fasta")


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

def load_plasmid_sequence(path: str | None = None) -> str:
    """Load the pBR322 nucleotide sequence (uppercase ATGC) from the FASTA."""
    path = path or _FASTA_PATH
    letters = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            letters.append(line.strip().upper())
    seq = "".join(letters)
    if not seq:
        raise ValueError(f"empty plasmid sequence in {path!r}")
    return seq


def _nt_mapping_from_dntps(dntps: list[str]) -> dict[str, int]:
    """Nucleotide letter → int index, in the SAME order the chromosome uses.

    The chromosome ``sequences`` were encoded from
    ``sim_data.dntp_code_to_id_ordered`` whose values are exactly
    ``molecule_groups.dntps`` (see molecule_groups.py). So the i-th dNTP id in
    the chromosome-replication config's ``dntps`` list carries the i-th
    nucleotide code. We recover the single-letter base from each id
    (``DATP[c]``→A, ``DCTP[c]``→C, ``DGTP[c]``→G, ``TTP[c]``→T) so the plasmid
    sequence is encoded with indices identical to the chromosome — required for
    the shared ``polymerize`` to see consistent bases.
    """
    mapping: dict[str, int] = {}
    for i, dntp_id in enumerate(dntps):
        base = dntp_id.split("[")[0]        # strip compartment tag "[c]"
        if base.startswith("D"):            # DATP/DCTP/DGTP → drop the deoxy 'D'
            base = base[1:]
        letter = base[0]                    # A / C / G / T
        mapping[letter] = i
    missing = {"A", "C", "G", "T"} - set(mapping)
    if missing:
        raise ValueError(
            f"could not derive nt mapping for {sorted(missing)} from dntps={dntps!r}")
    return mapping


def _complement(sequence_vector: np.ndarray) -> np.ndarray:
    """Vector complement of a DNA sequence in int-index form: (N-1) - v.

    With the ACGT→0123 index order this pairs A(0)↔T(3) and C(1)↔G(2).
    """
    return (_N_NT_TYPES - 1) - sequence_vector


def _build_plasmid_replication_sequences(
    plasmid_sequence: str,
    dntps: list[str],
    oriv_coordinate: int,
    buffer_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce commit 23137990's ``_build_plasmid_replication``.

    Derives the (2, maxLen) int8 ``plasmid_replication_sequences`` and the
    ``plasmid_replichore_lengths`` (single unidirectional replichore) from the
    plasmid sequence, the oriV coordinate, and the chromosome nt-index mapping.

    Returns:
        (plasmid_replication_sequences, plasmid_replichore_lengths)
    """
    nt_mapping = _nt_mapping_from_dntps(dntps)
    plasmid_length = len(plasmid_sequence)

    numerical_sequence = np.empty(plasmid_length, np.int8)
    for i, letter in enumerate(plasmid_sequence):
        numerical_sequence[i] = nt_mapping[letter]

    # Forward only, starting at oriV, wrapping around (unidirectional ColE1).
    order = np.hstack(
        (
            np.arange(oriv_coordinate, plasmid_length),
            np.arange(0, oriv_coordinate),
        )
    )
    plasmid_forward_sequence = numerical_sequence[order]
    plasmid_forward_complement_sequence = _complement(plasmid_forward_sequence)

    # Unidirectional: one replichore (not two like the chromosome).
    plasmid_replichore_lengths = np.array(
        [plasmid_forward_sequence.size], dtype=np.int64
    )

    max_len = np.int64(plasmid_forward_sequence.size + buffer_len)

    plasmid_replication_sequences = np.empty((2, max_len), np.int8)
    plasmid_replication_sequences.fill(polymerize.PAD_VALUE)
    plasmid_replication_sequences[
        0, : plasmid_forward_sequence.size
    ] = plasmid_forward_sequence
    plasmid_replication_sequences[
        1, : plasmid_forward_complement_sequence.size
    ] = plasmid_forward_complement_sequence

    return plasmid_replication_sequences, plasmid_replichore_lengths


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

# Brendel & Perelson 1993 (J Mol Biol 229:860-872) Table 1, pBR322 rom+
# parameterization. Rate constants in BP's native units (min⁻¹ for
# unimolecular; M⁻¹·min⁻¹ for the bimolecular k_1, k_3). PlasmidReplication's
# initialize() converts to per-second / per-count using V_c and Avogadro. BP
# Table 2 predicts ~28 copies/cell for rom+ wild-type with these values.
_BP1993_RNA_CONTROL = {
    "use_rna_control": True,
    "V_c_L": 6.25e-16,
    "n_avogadro": 6.022e23,
    "k_1": 1.5e8,
    "k_3": 1.7e8,
    "k_neg1": 48.0,
    "k_2": 44.0,
    "k_neg2": 0.085,
    "k_neg3": 0.17,
    "k_4": 34.0,
    "k_l": 12.0,
    "k_negl": 4.3,
    "k_p": 4.3,
    "k_D": 5.0,
    "k_negc": 17.0,
    "k_I": 6.0,
    "k_II": 0.25,
    "k_M": 4.0,
    "eps_I": 0.35,
    "eps_II": 0.35,
    "eps_M": 0.14,
    "n_substeps": 10,
}

# Parameters shared verbatim with chromosome replication — pulled straight from
# the baseline cache's ecoli-chromosome-replication config.
_SHARED_KEYS = (
    "polymerized_dntp_weights",
    "D_period",
    "replisome_protein_mass",
    "no_child_place_holder",
    "basal_elongation_rate",
    "make_elongation_rates",
    "mechanistic_replisome",
    "replisome_trimers_subunits",
    "replisome_monomers_subunits",
    "dntps",
    "ppi",
    "submass_indices",
)


def build_plasmid_replication_config(
    chromosome_config: dict[str, Any],
    seed: int = 0,
    plasmid_sequence: str | None = None,
    oriv_coordinate: int = ORIV_COORDINATE,
    time_step: int = 1,
) -> dict[str, Any]:
    """Assemble the full ``PlasmidReplication`` parameter dict (decoupled).

    Args:
        chromosome_config: the RESOLVED ``ecoli-chromosome-replication`` config
            (i.e. after ``resolve_config`` — ``make_elongation_rates`` is a
            callable, not a ``{'_function': ...}`` spec). Shared replication
            params are read from here.
        seed: per-process RNG seed.
        plasmid_sequence: override the pBR322 sequence (defaults to the FASTA).
        oriv_coordinate: oriV rotation origin (defaults to ``ORIV_COORDINATE``).
        time_step: process timestep.

    Returns:
        A parameter dict suitable for ``PlasmidReplication(config)``.
    """
    if plasmid_sequence is None:
        plasmid_sequence = load_plasmid_sequence()

    dntps = list(chromosome_config["dntps"])

    # Match the chromosome's polymerization-matrix buffer so buildSequences has
    # identical head-room per step.
    chrom_sequences = chromosome_config["sequences"]
    chrom_replichore = np.asarray(chromosome_config["replichore_lengths"])
    buffer_len = int(chrom_sequences.shape[1] - int(chrom_replichore.max()))
    if buffer_len <= 0:
        buffer_len = MAX_TIMESTEP * int(chromosome_config["basal_elongation_rate"])

    sequences, replichore_lengths = _build_plasmid_replication_sequences(
        plasmid_sequence, dntps, oriv_coordinate, buffer_len
    )

    config: dict[str, Any] = {
        "time_step": time_step,
        "seed": int(seed),
        "sequences": sequences,
        "replichore_lengths": replichore_lengths,
    }
    for key in _SHARED_KEYS:
        if key in chromosome_config:
            config[key] = chromosome_config[key]
    config.update(_BP1993_RNA_CONTROL)
    return config


# ---------------------------------------------------------------------------
# Initial unique-molecule state
# ---------------------------------------------------------------------------

def _plasmid_dna_mass_fg(
    plasmid_sequence: str,
    dntps: list[str],
    polymerized_dntp_weights: Any,
) -> float:
    """Double-stranded pBR322 DNA mass (fg) for the initial full_plasmid.

    ``polymerized_dntp_weights`` is a pint Quantity (grams / molecule) indexed
    like ``dntps``. Total mass = Σ over both strands of the per-nucleotide
    weight. For a duplex, each base on one strand is paired with its complement
    on the other, so the total composition is (fwd counts) + (fwd counts of the
    complementary base).
    """
    from v2ecoli.library.unit_bridge import unum_to_pint
    from v2ecoli.types.quantity import ureg as units

    nt_mapping = _nt_mapping_from_dntps(dntps)
    counts = np.zeros(_N_NT_TYPES, dtype=np.int64)
    for letter in plasmid_sequence:
        counts[nt_mapping[letter]] += 1
    # Complement strand: index (N-1)-i.
    counts_both = counts + counts[::-1]

    weights = unum_to_pint(polymerized_dntp_weights)
    weights_fg = np.asarray(weights.to(units.fg).magnitude, dtype=float)
    return float(np.dot(counts_both, weights_fg))


def _build_unique_array(
    template: np.ndarray,
    n_mols: int,
    unique_prefix: int,
    attrs: dict[str, Any],
):
    """Build a plasmid unique-molecule ``MetadataArray`` mirroring a template.

    ``template`` is the corresponding chromosome unique array (full_chromosome
    for full_plasmid, oriC for oriV, ...) — copying its dtype guarantees the
    structured layout the process-bigraph unique_array type expects. Attributes
    in ``attrs`` are written into the new rows; ``_entryState`` is set to 1;
    ``unique_index`` gets a distinct high-bit prefix (matching
    create_new_unique_molecules) and the array's ``.metadata`` is the next free
    index the runtime updater will hand out.
    """
    from v2ecoli.library.schema import MetadataArray

    arr = np.zeros(n_mols, dtype=template.dtype)
    for name, value in attrs.items():
        arr[name] = value
    if n_mols:
        arr["_entryState"] = 1
        arr["unique_index"] = np.arange(unique_prefix, unique_prefix + n_mols)
    return MetadataArray(arr, unique_prefix + n_mols)


def initial_plasmid_molecules(
    templates: dict[str, np.ndarray],
    dntps: list[str],
    polymerized_dntp_weights: Any,
    n_definitions: int,
    plasmid_sequence: str | None = None,
) -> dict[str, np.ndarray]:
    """Build the initial plasmid unique-molecule arrays.

    One full_plasmid (domain 0, carrying the pBR322 DNA mass), one oriV
    (domain 0), one plasmid_domain (domain 0, no children), and zero active
    replisomes — mirroring ``initialize_full_plasmid`` +
    ``determine_plasmid_state`` from commit 23137990, adapted to build the
    structured arrays directly from chromosome templates (no plasmid-aware
    sim_data required).

    Args:
        templates: mapping with the chromosome unique arrays to copy dtypes from
            — keys ``full_chromosome``, ``oriC``, ``chromosome_domain``,
            ``active_replisome``.
        dntps / polymerized_dntp_weights: from the chromosome-replication config.
        n_definitions: number of existing unique-molecule definitions; used to
            offset the plasmid unique-index prefixes so they never collide with
            chromosome molecules.
        plasmid_sequence: override the pBR322 sequence (defaults to the FASTA).

    Returns:
        Dict with ``full_plasmid``, ``oriV``, ``plasmid_domain``,
        ``plasmid_active_replisome`` MetadataArrays.
    """
    if plasmid_sequence is None:
        plasmid_sequence = load_plasmid_sequence()

    plasmid_mass = _plasmid_dna_mass_fg(
        plasmid_sequence, dntps, polymerized_dntp_weights
    )

    place_holder = -1
    # Distinct unique_index prefixes (index << 59), continuing past the existing
    # chromosome definitions so no two active molecules ever share an index.
    pfx = lambda k: (n_definitions + k) << 59  # noqa: E731

    full_plasmid = _build_unique_array(
        templates["full_chromosome"], 1, pfx(0),
        {
            "division_time": 0.0,
            "has_triggered_division": True,
            "domain_index": 0,
            "massDiff_DNA": plasmid_mass,
        },
    )
    oriV = _build_unique_array(
        templates["oriC"], 1, pfx(1),
        {"domain_index": 0},
    )
    plasmid_domain = _build_unique_array(
        templates["chromosome_domain"], 1, pfx(2),
        {
            "domain_index": 0,
            "child_domains": np.full((1, 2), place_holder, dtype=np.int32),
        },
    )
    plasmid_active_replisome = _build_unique_array(
        templates["active_replisome"], 0, pfx(3), {},
    )

    return {
        "full_plasmid": full_plasmid,
        "oriV": oriV,
        "plasmid_domain": plasmid_domain,
        "plasmid_active_replisome": plasmid_active_replisome,
    }


# Initial Brendel-Perelson ODE control state (one idle plasmid: D=1, one RNA I).
def initial_plasmid_rna_control() -> dict[str, Any]:
    """Initial ``plasmid_rna_control`` process_state (BP1993 species)."""
    return {
        "D": 1.0,
        "D_tII": 0.0,
        "D_lII": 0.0,
        "D_p": 0.0,
        "D_starc": 0.0,
        "D_c": 0.0,
        "D_M": 0.0,
        "R_I": 17.0,
        "R_II": 0.0,
        "M": 0.0,
        "repl_accum": 0.0,
        "n_rna_initiations": 0,
    }
