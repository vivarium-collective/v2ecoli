"""Flagella FlgM secretion — the Class II -> Class III timing gate.

Ported to process-bigraph / v2ecoli from Maya Abdalla's vEcoli ``biofilm`` branch
(``ecoli/processes/flagella_flgm_secretion.py``). Biology preserved verbatim; only
the framework scaffolding is adapted to ``EcoliStep``.

Mechanism
---------
Depletes cytoplasmic FlgM (G369-MONOMER[c]) at a rate proportional to the number
of complete flagella (CPLX0-7452[j]).

Once the hook-basal body is assembled and the filament added (forming CPLX0-7452,
the complete flagellum), the type-III secretion channel actively pumps cytoplasmic
FlgM out of the cell. As cytoplasmic FlgM drops, the FLGM-FLIA-CPLX equilibrium
(equilibrium_reactions.tsv) shifts toward releasing free FliA (EG11355-MONOMER[c]).
Free FliA then acts as sigma-28, activating Class III flagella promoters (fliC,
motAB, cheAW, ...) via :mod:`v2ecoli.processes.flagella_transcription_regulation`.

This process is the Class II -> Class III gate. Without it, FlgM accumulates
indefinitely and permanently sequesters FliA, preventing Class III expression.
With it, Class III genes activate only after flagellum assembly — producing the
timed transcriptional cascade seen experimentally (Kalir et al. 2001).

We use CPLX0-7452[j] (complete flagellum) rather than FLAGELLAR-MOTOR-COMPLEX[j]
because in a cell with assembled flagella the motor complex is consumed as a
subunit of CPLX0-7452; the latter is the persistent structure through which FlgM
is secreted at steady state.

Refs: Hughes KT et al. (1993) Science 262:1277; Aldridge PD & Hughes KT (2002)
Curr Opin Microbiol 5:160; Kalir S et al. (2001) Science 292:2080.

Ordered in the composite flow:
    ecoli-complexation -> ecoli-flagella-flgm-secretion -> ecoli-transcript-initiation
"""

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flagella-flgm-secretion"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flgm_secretion"),
    "global_time": ("global_time",),
}


class FlagellaFlgMSecretion(Step):
    """Secrete cytoplasmic FlgM through assembled flagella, releasing free FliA."""

    description = (
        "FlagellaFlgMSecretion — type-III secretion of the anti-sigma FlgM.\n\n"
        "    exported = min(FlgM, round(n_flagella * secretion_rate * timestep))\n"
        "  Depletes G369-MONOMER[c] proportional to complete flagella CPLX0-7452[j];\n"
        "  shifting the FlgM-FliA equilibrium to release sigma-28 and open Class III."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flgM_id": {"_type": "string", "_default": "G369-MONOMER[c]"},
        # CPLX0-7452 is the complete flagellum (hook-basal body + filament), [j].
        "hbb_id": {"_type": "string", "_default": "CPLX0-7452[j]"},
        # FlgM molecules exported per complete flagellum per second. Calibrated from
        # FlgM half-life measurements: t1/2 drops from ~30 min (no HBB) to ~2 min
        # (with HBB) -> net ~5.4e-3/s, ~0.1 molecules/flagellum/s across ~4 flagella.
        "secretion_rate": {"_type": "float", "_default": 0.1},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        self.secretion_rate = self.parameters["secretion_rate"]
        # Bulk indices resolved lazily against the live bulk array.
        self.flgM_idx = None
        self.hbb_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flgM_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flgM_idx = bulk_name_to_idx(self.parameters["flgM_id"], bulk_ids)
            self.hbb_idx = bulk_name_to_idx(self.parameters["hbb_id"], bulk_ids)

        hbb_count = counts(states["bulk"], self.hbb_idx)
        flgM_count = counts(states["bulk"], self.flgM_idx)

        exported = 0
        if hbb_count > 0 and flgM_count > 0:
            # secretion_rate molecules per HBB per second, scaled by timestep,
            # clamped to available FlgM so counts never go negative.
            exported = min(
                int(flgM_count),
                int(round(hbb_count * self.secretion_rate * states["timestep"])),
            )

        return {
            "bulk": [(self.flgM_idx, -exported)],
            "next_update_time": states["global_time"] + states["timestep"],
        }
