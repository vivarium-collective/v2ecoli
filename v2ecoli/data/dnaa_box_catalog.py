"""DnaA-box catalog for the dnaa-03 box-binding study.

Catalog covers every E. coli DnaA-binding site referenced in the
investigation expert document
``references/expert/replication_initiation_molecular_info.pdf``:

  - 11 oriC boxes (Katayama 2017, Kasho 2023)
        R1, R2, R4 — high-affinity, bind both DnaA-ATP and DnaA-ADP
        R5M, tau2, I1-3, C1-3 — low-affinity, DnaA-ATP preferential
  - 7 dnaA-promoter boxes (Speck 1999 + molecular_info doc revision)
        boxes 1, 2 — high, both forms
        box 3 — VERY low affinity (note: this is finer-grained than the
                "high/low" binary; modeled as "low" with form_pref="both" here)
        box 4 — ATP preferential
        box a — low, both
        box b, c — ATP preferential
  - 307 chromosomal consensus DnaA-boxes (TTWTNCACA; Roth 1998 + Olivi 2025).
        Coordinates here are synthetic (uniform spacing around the 4.6 Mb
        chromosome) since the Olivi 2025 supplementary positions are not yet
        in v2ecoli sim_data. v1 of dnaa-03 doesn't depend on the specific
        positions — only on the COUNT and the affinity-class distribution.
        Replace with the real coordinate list when the catalog is curated.

Also catalogued for completeness (inactive in this study; placeholders for
dnaa-05's RIDA / DDAH / DARS work):
  - datA: 4 DnaA boxes + 1 IBS  (Katayama 2017, Kasho 2023)
  - DARS1: 3 boxes              (Fujimitsu 2009)
  - DARS2: 3 boxes + IBS/FBS    (Kasho 2014)

Each entry has:
    box_id              str    — stable id, e.g. "oriC_R1", "dnaAp_box1",
                                   "chrom_consensus_42", "datA_box2"
    region_type         enum   — ORIC | DNAAP | DATA | DARS1 | DARS2 |
                                   CHROMOSOMAL_TITRATION
    position            int    — chromosomal coordinate (bp from oriC=0)
    affinity_class      enum   — high | low
    form_preference     enum   — both | atp_preferential
    cooperative_group   str    — null for independent, group id for sites
                                   that bind cooperatively (e.g.,
                                   "oric_right_filament" for R4 + C1 + C3 + I3 + C2)
    source              str    — bib_key of primary citation
    active_in_dnaa_03   bool   — True if v1 of dnaa-03 actuates this site.
                                   datA / DARS sites are False (dnaa-05).
    expert_reviewed     bool   — set True once an expert signs off.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List


CHROMOSOME_LENGTH_BP = 4_641_652  # E. coli K-12 MG1655 chromosome


@dataclasses.dataclass(frozen=True)
class DnaaBox:
    box_id: str
    region_type: str  # ORIC | DNAAP | DATA | DARS1 | DARS2 | CHROMOSOMAL_TITRATION
    position: int
    affinity_class: str  # high | low
    form_preference: str  # both | atp_preferential
    cooperative_group: str | None
    source: str
    active_in_dnaa_03: bool = True
    expert_reviewed: bool = False


# ─── 11 oriC boxes (Katayama 2017 + molecular_info doc) ────────────────────
# Positions are approximate within the 462-bp oriC region. Real coordinates
# depend on which oriC reference frame; v1 uses arbitrary monotone positions
# within oriC for now (binding kinetics don't care about exact bp).
_ORIC_BOXES: list[DnaaBox] = [
    DnaaBox("oriC_R1", "ORIC", 100, "high", "both", "oric_left_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_R2", "ORIC", 200, "high", "both", "oric_left_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_R4", "ORIC", 320, "high", "both", "oric_right_filament",
            "Katayama2017Frontiers"),
    # Low-affinity sites (ATP-preferential per molecular_info)
    DnaaBox("oriC_R5M",  "ORIC", 150, "low", "atp_preferential", "oric_left_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_tau2", "ORIC", 165, "low", "atp_preferential", "oric_left_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_I1",   "ORIC", 240, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_I2",   "ORIC", 260, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_I3",   "ORIC", 280, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_C1",   "ORIC", 300, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_C2",   "ORIC", 310, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
    DnaaBox("oriC_C3",   "ORIC", 315, "low", "atp_preferential", "oric_right_filament",
            "Katayama2017Frontiers"),
]


# ─── 7 dnaAp boxes (Speck 1999 + molecular_info doc) ───────────────────────
# Positions within the ~448-bp dnaA-promoter region.
_DNAAP_BOXES: list[DnaaBox] = [
    # Boxes 1, 2 — highest affinity, both forms
    DnaaBox("dnaAp_box1", "DNAAP", 3_882_500, "high", "both", None, "Speck1999EMBO"),
    DnaaBox("dnaAp_box2", "DNAAP", 3_882_540, "high", "both", None, "Speck1999EMBO"),
    # Box 3 — VERY low affinity (modeled as "low" with form_pref "both")
    DnaaBox("dnaAp_box3", "DNAAP", 3_882_560, "low", "both", None, "Speck1999EMBO"),
    # Box 4 — ATP preferential
    DnaaBox("dnaAp_box4", "DNAAP", 3_882_580, "low", "atp_preferential", None,
            "Speck1999EMBO"),
    # Boxes a, b, c — letters
    DnaaBox("dnaAp_boxa", "DNAAP", 3_882_600, "low", "both", None, "Speck1999EMBO"),
    DnaaBox("dnaAp_boxb", "DNAAP", 3_882_620, "low", "atp_preferential", None,
            "Speck1999EMBO"),
    DnaaBox("dnaAp_boxc", "DNAAP", 3_882_640, "low", "atp_preferential", None,
            "Speck1999EMBO"),
]


# ─── 307 chromosomal consensus boxes (TTWTNCACA) ────────────────────────────
# v1 uses uniform synthetic positions around the chromosome. Replace with
# the curated coordinate list from Olivi 2025 supplementary when available.
# All are high-affinity per molecular_info doc ("all 307 consensus DnaA
# boxes (TTWTNCACA) are high-affinity (Kd~1 nM)").
N_CHROMOSOMAL = 307


def _generate_chromosomal_boxes() -> list[DnaaBox]:
    """Synthetic uniform-spaced chromosomal box catalog (v1)."""
    out: list[DnaaBox] = []
    # Avoid the oriC region (0-462 bp) and dnaAp region (~3.88 Mb)
    step = CHROMOSOME_LENGTH_BP // N_CHROMOSOMAL
    for i in range(N_CHROMOSOMAL):
        pos = step * i + 5_000  # offset to avoid colliding with oriC
        out.append(DnaaBox(
            box_id=f"chrom_consensus_{i:03d}",
            region_type="CHROMOSOMAL_TITRATION",
            position=pos,
            affinity_class="high",     # per molecular_info: consensus = high
            form_preference="both",
            cooperative_group=None,
            source="Roth1998MolMicrobiol",
        ))
    return out


_CHROMOSOMAL_BOXES = _generate_chromosomal_boxes()


# ─── datA / DARS1 / DARS2 — inactive in dnaa-03, catalogued for dnaa-05 ────
_DATA_BOXES: list[DnaaBox] = [
    DnaaBox("datA_box2", "DATA", 4_400_000, "high", "both", "datA_essential",
            "Katayama2017Frontiers", active_in_dnaa_03=False),
    DnaaBox("datA_box3", "DATA", 4_400_050, "high", "both", "datA_essential",
            "Katayama2017Frontiers", active_in_dnaa_03=False),
    DnaaBox("datA_box7", "DATA", 4_400_100, "high", "both", "datA_essential",
            "Katayama2017Frontiers", active_in_dnaa_03=False),
    DnaaBox("datA_box4", "DATA", 4_400_150, "high", "both", "datA_stimulatory",
            "Katayama2017Frontiers", active_in_dnaa_03=False),
]
_DARS1_BOXES: list[DnaaBox] = [
    DnaaBox("dars1_I",   "DARS1", 3_500_000, "high", "both", "dars1_core",
            "Fujimitsu2009GenesDev", active_in_dnaa_03=False),
    DnaaBox("dars1_II",  "DARS1", 3_500_050, "high", "both", "dars1_core",
            "Fujimitsu2009GenesDev", active_in_dnaa_03=False),
    DnaaBox("dars1_III", "DARS1", 3_500_100, "high", "both", "dars1_core",
            "Fujimitsu2009GenesDev", active_in_dnaa_03=False),
]
_DARS2_BOXES: list[DnaaBox] = [
    DnaaBox("dars2_I",   "DARS2", 1_200_000, "high", "both", "dars2_core",
            "Kasho2014NAR", active_in_dnaa_03=False),
    DnaaBox("dars2_II",  "DARS2", 1_200_050, "high", "both", "dars2_core",
            "Kasho2014NAR", active_in_dnaa_03=False),
    DnaaBox("dars2_III", "DARS2", 1_200_100, "high", "both", "dars2_core",
            "Kasho2014NAR", active_in_dnaa_03=False),
]


# ─── Public catalog ─────────────────────────────────────────────────────────
ALL_BOXES: list[DnaaBox] = (
    _ORIC_BOXES + _DNAAP_BOXES + _CHROMOSOMAL_BOXES
    + _DATA_BOXES + _DARS1_BOXES + _DARS2_BOXES
)


def active_boxes() -> list[DnaaBox]:
    """Boxes the dnaa-03 binding Process operates on (excludes datA/DARS)."""
    return [b for b in ALL_BOXES if b.active_in_dnaa_03]


def boxes_by_region(region_type: str) -> list[DnaaBox]:
    return [b for b in ALL_BOXES if b.region_type == region_type]


def summarize() -> dict:
    by_region: dict[str, int] = {}
    by_affinity: dict[str, int] = {}
    for b in ALL_BOXES:
        by_region[b.region_type] = by_region.get(b.region_type, 0) + 1
        by_affinity[b.affinity_class] = by_affinity.get(b.affinity_class, 0) + 1
    return {
        "total_sites": len(ALL_BOXES),
        "active_in_dnaa_03": sum(1 for b in ALL_BOXES if b.active_in_dnaa_03),
        "by_region": by_region,
        "by_affinity": by_affinity,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(), indent=2))
