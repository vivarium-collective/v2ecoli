"""Molecular reference data for E. coli replication initiation.

Encodes the facts in ``docs/references/replication_initiation.md`` (and the
underlying PDF) as importable constants, so that:

  * Tests can assert that the codified data matches the curated source.
  * Reports and future processes can cite specific facts (DnaA box
    affinity class, datA chromosomal position, SeqA sequestration window,
    etc.) without re-typing them.

This module is data-only — no imports from process-bigraph or sim_data.
The current ``v2ecoli`` model does **not** yet consume these constants in
its initiation logic; closing that gap is the work tracked in the draft
PR for ``feat/replication-initiation-detail``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# DnaA box motifs (Schaper & Messer 1995; Hansen et al. 2006; Olivi 2025)
# ---------------------------------------------------------------------------

# Consensus 9-mer; W = A|T, N = any.
DNAA_BOX_CONSENSUS = 'TTWTNCACA'

# Relaxed bioinformatic motif from datA / DARS1 / DARS2 analyses.
# H = A|C|T, M = A|C, W = A|T, V = A|C|G.
DNAA_BOX_RELAXED_MOTIF = 'HHMTHCWVH'

# Highest-affinity exact 9-mer (Kd ~1 nM).
DNAA_BOX_HIGHEST_AFFINITY = 'TTATCCACA'

# Approximate dissociation constants.
DNAA_BOX_HIGH_AFFINITY_KD_NM: float = 1.0
DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND: float = 100.0


# ---------------------------------------------------------------------------
# Dataclasses for region descriptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DnaABox:
    """A named DnaA-binding site within a regulatory region.

    Attributes:
      name:               site label, e.g. 'R1', 'C3', 'box1', 'I'.
      affinity_class:     'high' (Kd ~1 nM) or 'low' (Kd > 100 nM).
      sequence:           exact 9-mer if known, else None.
      nucleotide_preference:
        'both'         — binds both DnaA-ATP and DnaA-ADP roughly equally
        'atp'          — DnaA-ATP-preferential
        'unspecified'  — not stated in the curated source
    """

    name: str
    affinity_class: str
    sequence: Optional[str] = None
    nucleotide_preference: str = 'unspecified'

    def __post_init__(self):
        if self.affinity_class not in {'high', 'low', 'very_low'}:
            raise ValueError(
                f'affinity_class must be high|low|very_low, '
                f'got {self.affinity_class!r}')
        if self.nucleotide_preference not in {'both', 'atp', 'unspecified'}:
            raise ValueError(
                f'nucleotide_preference must be both|atp|unspecified, '
                f'got {self.nucleotide_preference!r}')


@dataclass(frozen=True)
class IhfSite:
    """A named IHF (integration host factor) binding site."""

    name: str
    role: str  # e.g. 'primary', 'secondary'


@dataclass(frozen=True)
class OriCRegion:
    length_bp: int
    dnaA_boxes: tuple[DnaABox, ...]
    ihf_sites: tuple[IhfSite, ...]
    ordered_oligomerization_right_arm: tuple[str, ...]
    """Order in which DnaA-ATP loads onto low-affinity boxes on the right
    arm of the DOR, anchored by R4-bound DnaA. From the PDF: C1 → I3 → C2 → C3.
    """


@dataclass(frozen=True)
class DnaAPromoterRegion:
    length_bp: int
    promoters: tuple[str, ...]                 # e.g. ('p1', 'p2')
    p2_to_p1_strength_ratio: float             # ~3
    promoter_separation_bp: int                # ~80
    dnaA_boxes: tuple[DnaABox, ...]


@dataclass(frozen=True)
class DatARegion:
    length_bp: int
    chromosomal_position_min: float            # 94.7 min on E. coli chromosome
    n_dnaA_boxes: int
    essential_dnaA_box_names: tuple[str, ...]  # ('box2', 'box3', 'box7')
    stimulatory_dnaA_box_names: tuple[str, ...]
    n_ihf_sites: int


@dataclass(frozen=True)
class DarsRegion:
    name: str
    length_bp: int
    core_box_names: tuple[str, ...]            # boxes I, II, III
    extra_box_names: tuple[str, ...] = ()      # e.g. ('IV', 'V') for DARS2
    n_ihf_sites: int = 0
    n_fis_sites: int = 0
    is_dominant_in_vivo: bool = False


@dataclass(frozen=True)
class SeqASequestration:
    sequestration_window_minutes: float        # ~10 min
    fraction_of_doubling_time_at_rapid_growth: float  # ~1/3
    n_gatc_sites_oriC_lower_bound: int         # >10
    binds_state: str                           # 'hemimethylated'


@dataclass(frozen=True)
class Rida:
    """Regulatory inactivation of DnaA: clamp + Hda → DnaA-ATP hydrolysis."""

    clamp_protein: str                         # 'DnaN' (β-clamp)
    catalytic_partner: str                     # 'Hda'
    hda_nucleotide_state: str                  # 'ADP'
    hda_clamp_binding_motif_terminus: str      # 'N'
    reaction: str                              # 'DnaA-ATP -> DnaA-ADP'


# ---------------------------------------------------------------------------
# oriC (PDF figure: 462 bp; 11 DnaA boxes; 2 IHF sites)
# ---------------------------------------------------------------------------

ORIC = OriCRegion(
    length_bp=462,
    dnaA_boxes=(
        # 3 high-affinity sites — bind both ATP and ADP forms.
        DnaABox('R1', 'high', 'TTATCCACA', 'both'),
        DnaABox('R2', 'high', 'TTATACACA', 'both'),
        DnaABox('R4', 'high', 'TTATCCACA', 'both'),
        # 8 low-affinity sites — DnaA-ATP-preferential and cooperative.
        DnaABox('R5M', 'low', None, 'atp'),
        DnaABox('tau2', 'low', None, 'atp'),
        DnaABox('I1', 'low', None, 'atp'),
        DnaABox('I2', 'low', None, 'atp'),
        DnaABox('I3', 'low', None, 'atp'),
        DnaABox('C1', 'low', None, 'atp'),
        DnaABox('C2', 'low', None, 'atp'),
        DnaABox('C3', 'low', None, 'atp'),
    ),
    ihf_sites=(
        IhfSite('IBS1', 'primary'),
        IhfSite('IBS2', 'secondary'),
    ),
    ordered_oligomerization_right_arm=('C1', 'I3', 'C2', 'C3'),
)


# ---------------------------------------------------------------------------
# dnaA promoter (PDF figure: 448 bp; promoters p1, p2 ~80 bp apart)
# ---------------------------------------------------------------------------

DNAA_PROMOTER = DnaAPromoterRegion(
    length_bp=448,
    promoters=('p1', 'p2'),
    p2_to_p1_strength_ratio=3.0,
    promoter_separation_bp=80,
    dnaA_boxes=(
        DnaABox('box1', 'high', 'TTATCCACA', 'both'),
        DnaABox('box2', 'high', None, 'both'),
        DnaABox('box3', 'very_low', None, 'unspecified'),
        DnaABox('box4', 'low', None, 'atp'),     # overlaps box a
        DnaABox('boxa', 'low', None, 'unspecified'),
        DnaABox('boxb', 'low', None, 'atp'),
        DnaABox('boxc', 'low', None, 'atp'),
    ),
)


# ---------------------------------------------------------------------------
# datA (PDF figure: 363 bp; near oriC at 94.7 min)
# ---------------------------------------------------------------------------

DATA = DatARegion(
    length_bp=363,
    chromosomal_position_min=94.7,
    n_dnaA_boxes=4,
    essential_dnaA_box_names=('box2', 'box3', 'box7'),
    stimulatory_dnaA_box_names=('box4',),
    n_ihf_sites=1,
)


# ---------------------------------------------------------------------------
# DARS1 / DARS2
# ---------------------------------------------------------------------------

DARS1 = DarsRegion(
    name='DARS1',
    length_bp=632,
    core_box_names=('I', 'II', 'III'),
    extra_box_names=(),
    n_ihf_sites=0,
    n_fis_sites=0,
    is_dominant_in_vivo=False,
)

DARS2 = DarsRegion(
    name='DARS2',
    length_bp=737,
    core_box_names=('I', 'II', 'III'),
    extra_box_names=('IV', 'V'),
    n_ihf_sites=1,   # IBS1-2 (one named region in the PDF)
    n_fis_sites=2,   # FBS1 and FBS2-3
    is_dominant_in_vivo=True,
)


# ---------------------------------------------------------------------------
# SeqA / RIDA
# ---------------------------------------------------------------------------

SEQA = SeqASequestration(
    sequestration_window_minutes=10.0,
    fraction_of_doubling_time_at_rapid_growth=1.0 / 3.0,
    n_gatc_sites_oriC_lower_bound=10,
    binds_state='hemimethylated',
)

RIDA = Rida(
    clamp_protein='DnaN',
    catalytic_partner='Hda',
    hda_nucleotide_state='ADP',
    hda_clamp_binding_motif_terminus='N',
    reaction='DnaA-ATP -> DnaA-ADP',
)


# ---------------------------------------------------------------------------
# Bulk molecule IDs — EcoCyc identifiers used by sim_data and the
# equilibrium / metabolism / complexation processes.
# ---------------------------------------------------------------------------

# apo-DnaA monomer (no bound nucleotide). Source: ``flat/proteins.tsv``.
DNAA_APO_BULK_ID: str = 'PD03831[c]'

# DnaA-ATP complex. Equilibrium-bound from apo-DnaA + ATP via
# ``MONOMER0-160_RXN`` in ``flat/equilibrium_reactions.tsv``.
DNAA_ATP_BULK_ID: str = 'MONOMER0-160[c]'

# DnaA-ADP complex. Equilibrium-bound from apo-DnaA + ADP via
# ``MONOMER0-4565_RXN`` in ``flat/equilibrium_reactions.tsv``.
# Hydrolysis from DnaA-ATP via ``RXN0-7444`` (catalyzed by the
# Hda-β-clamp complex CPLX0-10342) — see ``flat/metabolic_reactions.tsv``.
DNAA_ADP_BULK_ID: str = 'MONOMER0-4565[c]'


# ---------------------------------------------------------------------------
# Region classifier (Phase 0)
# ---------------------------------------------------------------------------
#
# E. coli K-12 MG1655 reference (GenBank U00096.3). The model stores DnaA-box
# coordinates *relative to oriC*, computed by
# ``parca.reconstruction.ecoli.dataclasses.process.replication.
# _get_relative_coordinates``. We mirror that transform here so the
# classifier can accept the same relative coordinate the model uses.

GENOME_LENGTH_BP: int = 4_641_652
ORIC_ABS_CENTER_BP: int = 3_925_860   # midpoint of EcoCyc oriC site (3925744-3925975)
TERC_ABS_CENTER_BP: int = 1_609_168   # midpoint of EcoCyc terC site (1609157-1609179)


def _to_relative(abs_bp: int) -> int:
    """Convert an absolute MG1655 coordinate to relative-to-oriC.

    Mirrors the formula in
    ``parca...replication.Replication._get_relative_coordinates``."""
    rel = ((abs_bp - TERC_ABS_CENTER_BP) % GENOME_LENGTH_BP) \
        + TERC_ABS_CENTER_BP - ORIC_ABS_CENTER_BP
    if rel < 0:
        rel += 1
    return rel


# Absolute bp boundaries for each regulatory locus, centered on the EcoCyc
# minimal-site midpoint (or a literature-derived position for datA, which
# isn't a named EcoCyc site) and widened to the PDF region length so all
# named DnaA boxes — including non-consensus ones — fall inside the window.
# Window construction: ``lo = mid - width // 2``, ``hi = mid + (width-1) // 2``,
# so the inclusive width ``hi - lo + 1`` equals the PDF length exactly.
REGION_BOUNDARIES_ABS: dict[str, tuple[int, int]] = {
    'oriC':           (3_925_629, 3_926_090),  # 462 bp; ORIC.length_bp
    'dnaA_promoter':  (3_883_730, 3_884_177),  # 448 bp; upstream of dnaA CDS (-strand)
    'datA':           (4_396_023, 4_396_385),  # 363 bp; near 94.7 min on the chromosome
    'DARS1':          (   812_808,    813_439),  # 632 bp; centered on EcoCyc DARS1
    'DARS2':          (2_968_784, 2_969_520),  # 737 bp; centered on EcoCyc DARS2
}


def _abs_pair_to_rel(lo_abs: int, hi_abs: int) -> tuple[int, int]:
    lo, hi = _to_relative(lo_abs), _to_relative(hi_abs)
    return (lo, hi) if lo <= hi else (hi, lo)


# Pre-computed relative-coord boundaries (the form the model uses on the
# DnaA_box ``coordinates`` field).
REGION_BOUNDARIES: dict[str, tuple[int, int]] = {
    name: _abs_pair_to_rel(lo, hi)
    for name, (lo, hi) in REGION_BOUNDARIES_ABS.items()
}


def region_for_coord(rel_bp: int) -> Optional[str]:
    """Classify a DnaA-box relative coordinate by regulatory region.

    Returns the region name (``'oriC'``, ``'dnaA_promoter'``, ``'datA'``,
    ``'DARS1'``, ``'DARS2'``) when ``rel_bp`` falls inside a known
    regulatory locus; ``None`` when the coordinate is elsewhere on the
    chromosome.

    The argument is a coordinate *relative to oriC*, matching the format
    of ``DnaA_box.coordinates`` in the simulation state.
    """
    for name, (lo, hi) in REGION_BOUNDARIES.items():
        if lo <= rel_bp <= hi:
            return name
    return None


# Empirical baseline (distinct coordinates): the bioinformatic strict-
# consensus search (``DnaA_box`` motif = TTWTNCACA, 8 sequence variants;
# see ``flat/sequence_motifs.tsv``) finds these counts of distinct
# coordinates per region when ``region_for_coord`` is applied to the
# init-state DnaA-box coordinates. The total (8 distinct) is far less
# than the 30 PDF-named boxes — the strict motif misses named low-
# affinity non-consensus boxes. Phase 2 (DnaA-box binding) closes this
# gap by enriching the box list with the named non-consensus boxes.
PER_REGION_STRICT_CONSENSUS_COUNT: dict[str, int] = {
    'oriC':          3,
    'dnaA_promoter': 1,
    'datA':          0,
    'DARS1':         1,
    'DARS2':         3,
}

# Curated counts from the reference PDF (named boxes per region — high- and
# low-affinity sites, including non-consensus ones). The ratio
# ``PER_REGION_STRICT_CONSENSUS_COUNT[r] / PER_REGION_PDF_COUNT[r]`` is the
# coverage of bioinformatic search for that locus.
PER_REGION_PDF_COUNT: dict[str, int] = {
    'oriC':          len(ORIC.dnaA_boxes),
    'dnaA_promoter': len(DNAA_PROMOTER.dnaA_boxes),
    'datA':          DATA.n_dnaA_boxes,
    'DARS1':         len(DARS1.core_box_names) + len(DARS1.extra_box_names),
    'DARS2':         len(DARS2.core_box_names) + len(DARS2.extra_box_names),
}
