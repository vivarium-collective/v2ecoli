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
