"""Flagella FlgM secretion — the Class II -> Class III timing gate.

Ported to process-bigraph / v2ecoli from Maya Abdalla's vEcoli ``biofilm`` branch
(``ecoli/processes/flagella_flgm_secretion.py``). Biology preserved verbatim; only
the framework scaffolding is adapted to ``EcoliStep``.

Mechanism
---------
Depletes cytoplasmic FlgM (G369-MONOMER[c]) at a rate proportional to the number
of completed hook-basal bodies -- see "TRIGGER FIX" below for exactly what that
means and why it changed.

Once the hook-basal body is assembled, the type-III secretion channel actively
pumps cytoplasmic FlgM out of the cell. As cytoplasmic FlgM drops, the
FLGM-FLIA-CPLX equilibrium (equilibrium_reactions.tsv) shifts toward releasing
free FliA (EG11355-MONOMER[c]). Free FliA then acts as sigma-28, activating
Class III flagella promoters (fliC, motAB, cheAW, ...) via
:mod:`v2ecoli.processes.flagella_transcription_regulation`.

This process is the Class II -> Class III gate. Without it, FlgM accumulates
indefinitely and permanently sequesters FliA, preventing Class III expression.
With it, Class III genes activate only after hook-basal-body assembly —
producing the timed transcriptional cascade seen experimentally (Kalir et al.
2001).

TRIGGER FIX (2026-08-25, REVISED 2026-08-27): previously, this Step counted
ONLY CPLX0-7452[j] -- the FULLY complete flagellum, hook-basal body *plus
filament* -- as the secretion trigger. That's biologically wrong. Hughes KT et
al. (1993) Science 262:1277 ("Sensing structural intermediates in bacterial
flagellar assembly by export of a negative regulator") and Karlinsey JE et al.
(2000) Mol Microbiol 37:1220 ("Completion of the hook-basal body complex... is
coupled to FlgM secretion and fliC transcription") both establish that the
substrate-specificity switch (FliK/FlhB-mediated) that starts FlgM export
happens as soon as the hook-basal body itself is done -- well BEFORE the
filament is built. The filament can take multiple cell-division cycles to
complete at this model's target_length=5000 (see
flagella_filament_elongation.py's docstring), so gating secretion on
CPLX0-7452 alone meant the negative feedback almost never engaged: confirmed
directly via a population test where "complete flagella" (CPLX0-7452) grew
only 4->5->6 over 90 simulated minutes while free FliA grew unchecked from
~250 to ~17,000 over the same window.

The real trigger species already exists in this model:
flagella_nfsim_complexation.py creates a `nascent_flagellum` unique molecule at
the EXACT moment its internal 'flagella' species forms. That species is
confirmed (generate_flagella_bngl.py's own docstring, "assembly complete
through the HOOK-BASAL-BODY stage only (motor + hook + hook-filament junction +
cap machinery)") to BE the hook-basal-body-complete stage, nothing more,
nothing less -- so count(nascent_flagellum) alone is exactly "HBB is done."

FIRST ATTEMPT (2026-08-25, REVERTED): counted count(nascent_flagellum) +
count(CPLX0-7452), reasoning that a fully complete flagellum (filament also
done) should keep contributing to secretion since its export apparatus doesn't
physically disappear -- CPLX0-7452 "consumes the motor complex as a subunit"
per this file's own note below, so the channel is still there. That's a
plausible mechanism, but "secretion continues at the same rate indefinitely
after the filament also finishes" was only ever backed by an unverified,
websearch-summarized characterization, not a primary source read directly --
unlike the HBB-triggers-secretion mechanism itself, which IS directly confirmed
in Hughes (1993) and Karlinsey (2000). Given that gap, and given this
additive trigger caused a NEW division-crash regression (equilibrium solver
"Negative values" failure on FLGM-FLIA-CPLX_RXN, confirmed via a clean A/B:
identical seed/cache that ran cleanly before this change), reverted to the
more conservative, fully-confirmed reading: nascent_flagellum only.

We still don't use FLAGELLAR-MOTOR-COMPLEX[j] for the trigger even though it's
technically the last real bulk species before hook-basal-body completion,
because in a cell with assembled flagella the motor complex is consumed as a
subunit of CPLX0-7452, so it's not a stable, independently-countable species by
the time the HBB stage this Step cares about is reached -- nascent_flagellum
(the unique molecule created at that exact moment) is the only unambiguous
signal.

Refs: Hughes KT et al. (1993) Science 262:1277; Karlinsey JE et al. (2000) Mol
Microbiol 37:1220; Aldridge PD & Hughes KT (2002) Curr Opin Microbiol 5:160;
Kalir S et al. (2001) Science 292:2080.

Ordered in the composite flow:
    ecoli-complexation -> ecoli-flagella-flgm-secretion -> ecoli-transcript-initiation
"""


from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


NAME = "ecoli-flagella-flgm-secretion"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flgm_secretion"),
    "global_time": ("global_time",),
}


class FlagellaFlgMSecretion(Step):
    """Secrete cytoplasmic FlgM through completed hook-basal bodies, releasing
    free FliA. See module docstring's "TRIGGER FIX" for why the trigger is
    count(nascent_flagellum) alone -- confirmed to BE the hook-basal-body-
    complete stage, not an approximation of it."""

    description = (
        "FlagellaFlgMSecretion — type-III secretion of the anti-sigma FlgM.\n\n"
        "    n_hbb = count(nascent_flagellum)\n"
        "    exported = min(FlgM, round(n_hbb * secretion_rate * timestep))\n"
        "  Depletes G369-MONOMER[c] proportional to completed hook-basal bodies;\n"
        "  shifting the FlgM-FliA equilibrium to release sigma-28 and open Class III."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flgM_id": {"_type": "string", "_default": "G369-MONOMER[c]"},
        # CPLX0-7452 (FULLY complete flagellum, hook-basal body + filament) is
        # NO LONGER used by update() as of 2026-08-27 -- see module docstring's
        # "TRIGGER FIX" for the full history (tried adding it in, reverted).
        # Kept here, unused, only so the parameter isn't silently orphaned for
        # anyone re-checking this decision later.
        "hbb_id": {"_type": "string", "_default": "CPLX0-7452[j]"},
        # FlgM molecules exported per completed hook-basal body per second.
        # UNVERIFIED (2026-08-27): this comment originally claimed calibration
        # from "FlgM half-life measurements: t1/2 drops from ~30 min (no HBB)
        # to ~2 min (with HBB)" with no citation attached. Direct primary-
        # source check (Karlinsey JE et al. 1998, J Bacteriol 180:5384, Results
        # section) found DIFFERENT real numbers instead: Fla+ (HBB-complete)
        # cells show FlgM half-life = 7.3 min (strain TH2592), while HBB-
        # incomplete ring mutants (flgB, DeltaflgHI) show NO DETECTABLE
        # TURNOVER AT ALL -- not a slow ~30 min baseline. secretion_rate=0.1 has
        # NOT yet been re-derived from the confirmed real number; treat this
        # value as a placeholder pending that recalibration, not a settled one.
        "secretion_rate": {"_type": "float", "_default": 0.1},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "nascent_flagellum": {"_type": NASCENT_FLAGELLUM_ARRAY, "_default": []},
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
        # hbb_idx no longer resolved/used as of 2026-08-27 (see config_schema
        # note on hbb_id) -- kept at None, not deleted, per standing
        # preserve-old-code rule.
        self.hbb_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flgM_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flgM_idx = bulk_name_to_idx(self.parameters["flgM_id"], bulk_ids)
            # self.hbb_idx = bulk_name_to_idx(self.parameters["hbb_id"], bulk_ids)
            # ^ no longer resolved as of 2026-08-27 -- CPLX0-7452 is not part
            # of the trigger anymore, see module docstring's "TRIGGER FIX".

        # Old trigger (pre-2026-08-25 fix), kept per standing preserve-old-code
        # rule: counted ONLY fully-complete flagella (CPLX0-7452, hook-basal
        # body + filament). Biologically wrong -- the real substrate-
        # specificity switch that starts FlgM export happens at hook-basal-
        # body completion alone, well before the filament (which can take
        # multiple division cycles) finishes.
        # hbb_count = counts(states["bulk"], self.hbb_idx)

        # Intermediate version (2026-08-25, REVERTED 2026-08-27), kept per
        # standing preserve-old-code rule: added CPLX0-7452's count back in on
        # top of nascent_flagellum, reasoning that a fully complete flagellum
        # should keep secreting since its export apparatus doesn't disappear.
        # Reverted -- see module docstring's "TRIGGER FIX" -- that additive
        # assumption wasn't backed by a primary source the way the trigger
        # event itself is, and it caused a new division-crash regression.
        # (nascent_lengths,) = attrs(states["nascent_flagellum"], ["filament_length"])
        # hbb_count = counts(states["bulk"], self.hbb_idx) + nascent_lengths.size

        # Current (2026-08-27): nascent_flagellum alone. Confirmed (see module
        # docstring) to BE the hook-basal-body-complete stage exactly --
        # created the instant flagella_nfsim_complexation.py's internal
        # 'flagella' species forms, before any filament growth starts.
        (nascent_lengths,) = attrs(states["nascent_flagellum"], ["filament_length"])
        hbb_count = nascent_lengths.size
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
