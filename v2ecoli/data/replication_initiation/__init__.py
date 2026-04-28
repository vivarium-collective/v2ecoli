"""Curated molecular reference data for replication initiation.

The constants exported here come from
``docs/references/replication_initiation.md`` and the underlying PDF
``docs/references/replication_initiation_molecular_info.pdf``. They are the
single source of truth that ``tests/test_replication_initiation_reference.py``
asserts against. Update both the data and the doc together.
"""

from v2ecoli.data.replication_initiation.molecular_reference import (
    DNAA_BOX_CONSENSUS,
    DNAA_BOX_RELAXED_MOTIF,
    DNAA_BOX_HIGHEST_AFFINITY,
    DNAA_BOX_HIGH_AFFINITY_KD_NM,
    DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND,
    DnaABox,
    IhfSite,
    OriCRegion,
    DnaAPromoterRegion,
    DatARegion,
    DarsRegion,
    SeqASequestration,
    Rida,
    ORIC,
    DNAA_PROMOTER,
    DATA,
    DARS1,
    DARS2,
    SEQA,
    RIDA,
    GENOME_LENGTH_BP,
    ORIC_ABS_CENTER_BP,
    TERC_ABS_CENTER_BP,
    REGION_BOUNDARIES,
    REGION_BOUNDARIES_ABS,
    PER_REGION_STRICT_CONSENSUS_COUNT,
    PER_REGION_PDF_COUNT,
    region_for_coord,
)

__all__ = [
    'DNAA_BOX_CONSENSUS',
    'DNAA_BOX_RELAXED_MOTIF',
    'DNAA_BOX_HIGHEST_AFFINITY',
    'DNAA_BOX_HIGH_AFFINITY_KD_NM',
    'DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND',
    'DnaABox',
    'IhfSite',
    'OriCRegion',
    'DnaAPromoterRegion',
    'DatARegion',
    'DarsRegion',
    'SeqASequestration',
    'Rida',
    'ORIC',
    'DNAA_PROMOTER',
    'DATA',
    'DARS1',
    'DARS2',
    'SEQA',
    'RIDA',
    'GENOME_LENGTH_BP',
    'ORIC_ABS_CENTER_BP',
    'TERC_ABS_CENTER_BP',
    'REGION_BOUNDARIES',
    'REGION_BOUNDARIES_ABS',
    'PER_REGION_STRICT_CONSENSUS_COUNT',
    'PER_REGION_PDF_COUNT',
    'region_for_coord',
]
