"""Instantaneous chromosomal locus copy number from replication-fork positions.

A locus at oriC-relative coordinate ``c`` (signed: >0 = right replichore, <0 =
left replichore) starts at 1 copy and is duplicated each time a replication fork
on its own replichore passes it. So at any tick:

    copy_number(c) = 1 + (# active replisomes on c's replichore with |fork| >= |c|)

This gives the gene-dosage effect — loci near oriC (small |c|) are duplicated
early and spend more of the cycle at higher copy number than loci near terC.
Used by the extrinsic DnaA-conversion machinery (DDAH at datA, DARS1/DARS2),
whose activities scale with their locus copy number.

oriC-relative coordinates of the DnaA-cycle loci (computed from the MG1655
absolute genome coordinates with oriC at ~3,925,744 bp; verified against the
model's dnaA-promoter box position):

    DARS1 : +1_529_044   (right replichore, ~66% to terC)
    DARS2 :   -956_504   (left replichore,  ~41% to terC)
    datA  :   +467_079   (right replichore, ~20% to terC)
"""

from __future__ import annotations

import numpy as np

DARS1_COORD = 1_529_044
DARS2_COORD = -956_504
DATA_COORD = 467_079


def locus_copy_number(fork_coordinates, locus_coord: float) -> int:
    """Copy number of the locus at ``locus_coord`` given current fork positions.

    Args:
        fork_coordinates: signed oriC-relative coordinates of active replisomes
            (e.g. attrs(states["active_replisomes"], ["coordinates"])).
        locus_coord: signed oriC-relative coordinate of the locus.

    Returns:
        Integer copy number (>= 1).
    """
    fc = np.asarray(fork_coordinates, dtype=float)
    if fc.size == 0:
        return 1
    # same replichore (same sign; product >= 0 keeps the c==0 / fork==0 edge cases)
    same_side = (fc * locus_coord) >= 0
    passed = np.abs(fc) >= abs(locus_coord)
    return int(1 + np.sum(same_side & passed))
