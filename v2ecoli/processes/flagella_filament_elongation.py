"""Flagellar filament elongation — incremental FliC polymerization, and why
this couldn't be a single Gillespie complexation reaction.

Added 2026-08-06, part of Maya Abdalla's flagella-cascade investigation.

The problem this replaces
--------------------------
A real flagellar filament is built from ~20,000 FliC (flagellin) subunits
(PMC7696725: "The bacterial flagellar filament consists of approximately
20,000 flagellins and can be several micrometers long"). The original
stoichiometry fix for CPLX0-7452_RXN (complexation_reactions_modified.tsv)
correctly set FliC's coefficient to -20000 -- but a full ParCa rebuild with
that in place hung for 55+ minutes and had to be killed. Direct
investigation (macOS `sample` on the live process, plus `strings` on the
compiled extension) traced it to stochastic_arrow/arrowhead
(the Gillespie SSA engine behind ecoli-complexation): astronomically large
values (~1e19, ~1e35) were being computed as REACTION PROPENSITIES, not
mass-balance sums. Gillespie SSA propensities involve combinatorial terms
on reactant counts (roughly "count choose stoichiometric-coefficient") --
trivial for ordinary reactions needing a few copies, but combinatorially
explosive for a single discrete event needing 20,000 copies of the same
molecule at once. This is a known, fundamental limitation of stochastic
simulation algorithms for high-order reactions, not a matter of the solver
just needing more time.

It's also not how real biology does it: a filament isn't built in one
instantaneous event. FliC subunits are exported one at a time through the
hollow channel and added at the distal tip, incrementally, over minutes.
CPLX0-7452_RXN's -20000 coefficient for FliC is KEPT in
complexation_reactions_modified.tsv (needed for ParCa's static mass/
compartment auto-derivation -- ordinary arithmetic, not combinatorics, so
it's fine there) but is EXCLUDED from the runtime Gillespie config (see
get_complexation_config in sim_data.py) so ecoli-complexation never tries
to fire it. This Step (and flagella_filament_nucleation.py) handle the real,
incremental version instead, exactly mirroring how this codebase already
treats ribosomal translation (active_ribosome/peptide_length,
polypeptide_elongation.py) rather than as one giant complexation event.

Mechanism and rate
-------------------
Growth rate is length-dependent, not constant -- diffusion of subunits
through the growing channel becomes rate-limiting as the filament gets
longer (the "injection-diffusion" model): Renault et al. 2017, eLife
6:e23136, "Bacterial flagella grow through an injection-diffusion
mechanism." That paper gives an explicit growth-rate formula,
dL/dt = a / (b + L). rate_a=26,450, rate_b=575 subunits are KEPT as-is
(2026-08-21 review) -- see the CITATION AUDIT note below for why this
was investigated but deliberately not changed.

CITATION AUDIT (2026-08-21): the original derivation of rate_a=26,450
contained a real arithmetic error (it claimed "83-100 nm/min *
2130 subunits/um / 60s ~= 42-50 subunits/s at L=0", but that conversion
actually works out to ~2.95-3.55 subunits/s -- independently confirmed via
the paper's separate ~1,700 amino acids/s figure / ~500 aa per flagellin
subunit ~= ~3.4 subunits/s). Investigating a fix surfaced a bigger problem:
the paper's own reported numbers do NOT converge on one consistent "a"
under this simple model, no matter which two you combine to solve for the
third:
  - initial rate (~83-100 nm/min, i.e. ~3-3.5 subunits/s, robustly
    confirmed two independent ways) + b=575 subunits => a ~= 1,870
    subunit^2/s. This predicts ~35 HOURS to reach the paper's own
    ~21,300-subunit (~10 um) benchmark -- not necessarily wrong (the
    paper's "over 180+ minutes" phrasing is an explicit LOWER BOUND from a
    finite real-time-imaging session, not a claimed completion time, so
    there's no actual contradiction here), but far too slow for this
    codebase's practical simulation windows.
  - total growth time (~180+ min to ~21,300 subunits) + b=575 subunits =>
    a ~= 22,100 subunit^2/s -- implies an initial rate of ~1,080 nm/min,
    over 10x faster than the paper's own directly-reported ~83-100 nm/min.
  - a single-source "a ~= 0.2 um^2/min" figure (found once, could not be
    independently verified past a paywall) converts to a ~= 15,100
    subunit^2/s -- doesn't reconcile with either of the above either.
  None of these is clearly "the" right answer; the paper reports its
  actual fit in kon/diffusion-coefficient terms, not this a/b
  parameterization, so any single-a,b reduction is already an
  approximation with no unique inverse from summary statistics alone.

DECISION (2026-08-21, Maya's explicit call): kept rate_a=26,450 rather
than switching to any of the alternatives above. The literature doesn't
resolve to a single defensible value, and the practically-relevant
alternatives are all much SLOWER (1,870-22,100 vs 26,450) -- switching
would reopen the exact "doesn't complete within a practical simulation
window" problem that motivated cutting target_length from 20,000 down to
10,000 then to 5,000 in the first place (see below). This is a documented
open item, not a resolved citation -- a future, more careful re-derivation
directly from the paper's kon/D fit (not from its summary-statistic
sentences) would be the right way to actually settle this, not another
back-of-envelope pass.

When multiple nascent filaments exist simultaneously, they compete for the
same free FliC pool: desired increments are computed per filament, then
scaled down proportionally (fair-share) if their sum exceeds what's
actually available, so total consumption never exceeds free FliC.

Completion: once filament_length reaches the target (matching CPLX0-7452_RXN's
real FliC coefficient exactly, so total mass is conserved by construction --
see internal_state.py's nascent_flagellum registration), the Step consumes
5x FliD (cap, matching CPLX0-7452_RXN's FliD coefficient), deletes the
nascent_flagellum unique molecule, and increments the real CPLX0-7452 bulk
count by 1 -- the same count every other flagella_regulation Step reads.

Target length changed 2026-08-10 from 20,000 to 10,000 subunits. Both are
real, cited values, not an arbitrary diagnostic override -- the literature
range for filament length is ~20,000-40,000 subunits (5-20 um, at
~2,000-2,130 subunits/um), so 10,000 (~5 um) is the short end of that real
range, not an invented number. Chosen specifically because completion time
scales roughly with L^2/a for L>>b (dL/dt=a/(b+L) means T~=(bL+L^2/2)/a),
so this isn't a proportional time saving: 20,000->10,000 cuts minimum
completion time from ~133 min to ~35 min, making single-generation
completion achievable within this investigation's practical simulation
windows. Kept in sync with CPLX0-7452_RXN's FliC coefficient in
complexation_reactions_modified.tsv (also changed to -10,000, old value
kept as a comment there) so ParCa's own recorded molecular weight for a
complete flagellum matches what elongation actually builds.

Target length changed AGAIN 2026-08-11 from 10,000 to 5,000 subunits (~2.5
um) -- still a real, cited value on the short end of the 20,000-40,000
range (PMC7696725), not arbitrary. Motivated by direct evidence at 10,000:
the single-gen panel run showed free FliC dropping 51,967 -> 14 over 2400s
with only 4 concurrent nascent flagella and the longest filament only 85%
complete (8,540/10,000) -- the pool was close to fully exhausting itself
before even one flagellum finished. Halving again roughly quarters minimum
completion time (L^2 scaling) and restores real headroom in the FliC pool.
Kept in sync with CPLX0-7452_RXN's FliC coefficient in
complexation_reactions_modified.tsv (also changed to -5,000, old value kept
as a comment there).

Ordered in the composite flow: after ecoli-flagella-nfsim-complexation (the
NFsim assembly Step that creates nascent_flagellum entries -- see that
module's docstring). Corrected 2026-08-24: this previously said "after
ecoli-flagella-filament-nucleation," the deterministic Step this NFsim
pipeline replaced and archived 2026-08-21 (see
archive/deterministic-flagella-assembly-2026-08/) -- stale after the
switch, since that Step no longer runs at all. This ordering is enforced
by data dependency, not just Step sequence: this Step operates on
whatever's in the nascent_flagellum array, which starts empty and is
populated only by the upstream assembly Step, so it's a real no-op
(returns immediately, see update()'s n_active==0 check) until that
happens, regardless of which specific upstream Step is wired in.

FliS chaperone recycling (added 2026-08-21)
--------------------------------------------
FliS is a real flagellin export chaperone (binds FliC 1:1, escorts it to the
FlhA export gate) that is RELEASED and RECYCLED to bind a new FliC molecule
once export completes. 1:1 binding affinity is Muskotal et al. 2006 (FEBS
Lett 580:3916, isothermal titration calorimetry, Ka=1.9e7/M,
Kd~=5.26e-8M -- the same number used in FLIS-FLIC-CPLX_RXN's rate). The
release-and-recycling itself is directly demonstrated by the
FliS/flagellin/FliW heterotrimer structure (Scientific Reports 8:11115,
2018: "FliW and FliS are released during flagellin export... After
release, FliW and FliS are recycled").

CORRECTION (2026-08-21): a previous version of this docstring also cited
Evans, Stafford, Ahmed, Fraser & Hughes 2006 (PNAS 103:17474, "An escort
mechanism for cycling of export chaperones during flagellum assembly") as
direct support for FliS/FliC recycling specifically. That paper is real,
but its actual finding is FliJ-mediated escort recycling for the MINOR
filament-class subunit chaperones -- FliT (cap/FliD) and FlgN (hook-
filament junction/FlgK,FlgL) -- not FliS/FliC. Kept only as general
precedent that chaperone cycling at the export gate is a real mechanism in
this pathway, not as direct evidence for FliS itself; Sci Rep 8:11115 above
is the actual direct citation for what this Step implements.

This Step previously consumed free FliC directly and never touched
FLIS-FLIC-CPLX at all, so the complex just sat there governed only by the
passive Kd equilibrium -- no directed "export releases the chaperone"
event, meaning a real, literature-confirmed catalytic cycle was entirely
missing. Investigated 2026-08-21 after
observing free FliS crash from its ambient ~2,000 to near-zero within
minutes of a population run and never recover (confirmed this was NOT a
transcription-regulation gap -- fliS already shares fliD's real Class III
promoter/TU, see get_flagella_transcription_regulation_config's own note --
and NOT a translation-efficiency gap either, both genes have identical
translation_efficiencies_by_monomer).

Fix: elongation now draws preferentially from FLIS-FLIC-CPLX (protected
FliC) first, releasing one free FliS per unit consumed from it (the
chaperone returning to the cytoplasmic pool, exactly as the real escort
cycle does), and only falls back to free (unprotected) FliC for any
remaining demand. Fair-share scaling (for multiple simultaneous filaments)
is computed against the COMBINED available pool (protected + free), not
free FliC alone. This lets a small, real-abundance FliS pool protect a much
larger CUMULATIVE amount of FliC over time via repeated cycling, instead of
being capped at protecting only its own ambient count at any one instant.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


NAME = "ecoli-flagella-filament-elongation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_filament_elongation"),
    "global_time": ("global_time",),
}

# TARGET_LENGTH = 10000  # changed 2026-08-10 from 20000 -- kept per standing
                          # preserve-old-code rule, see module docstring
TARGET_LENGTH = 5000   # subunits; matches CPLX0-7452_RXN's FliC coefficient
                        # changed 2026-08-11 from 10000 -- see module docstring


class FlagellaFilamentElongation(Step):
    """Incremental FliC polymerization onto nascent flagella."""

    description = (
        "FlagellaFilamentElongation — length-dependent incremental FliC addition.\n\n"
        "    rate(L) = a / (b + L)   [subunits/s], a~=26450, b~=575 (Renault et al. 2017)\n"
        "  Grows each nascent_flagellum's filament_length; fair-shares free FliC across\n"
        "  simultaneous filaments; on reaching target_length (10,000, real short-range\n"
        "  subunit count as of 2026-08-10), consumes\n"
        "  5x FliD and converts the unique molecule into +1 real CPLX0-7452 bulk count."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "fliC_id": {"_type": "string", "_default": "EG10321-MONOMER[e]"},
        "fliD_id": {"_type": "string", "_default": "EG10841-MONOMER[e]"},
        "flagellum_id": {"_type": "string", "_default": "CPLX0-7452[j]"},
        "fliD_per_completion": {"_type": "integer", "_default": 5},
        "target_length": {"_type": "integer", "_default": TARGET_LENGTH},
        # Renault et al. 2017 (eLife 6:e23136) injection-diffusion model,
        # converted to subunit-count units -- see module docstring's
        # "CITATION AUDIT" / "DECISION" notes (2026-08-21): a real
        # arithmetic error was found in the original derivation, but the
        # paper's numbers don't converge on one clean replacement value
        # either, and every candidate alternative is much slower -- kept
        # at 26,450 deliberately (Maya's call), not re-derived.
        "rate_a": {"_type": "float", "_default": 26450.0},
        "rate_b": {"_type": "float", "_default": 575.0},
        # FliS chaperone recycling (added 2026-08-21, see module docstring).
        "fliS_id": {"_type": "string", "_default": "EG11388-MONOMER[c]"},
        "flis_flic_cplx_id": {"_type": "string", "_default": "FLIS-FLIC-CPLX[e]"},
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
            "nascent_flagellum": NASCENT_FLAGELLUM_ARRAY,
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        self.fliD_per_completion = self.parameters["fliD_per_completion"]
        self.target_length = self.parameters["target_length"]
        self.rate_a = self.parameters["rate_a"]
        self.rate_b = self.parameters["rate_b"]
        self.fliC_idx = None
        self.fliD_idx = None
        self.flagellum_idx = None
        self.fliS_idx = None
        self.flis_flic_cplx_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.fliC_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.fliC_idx = bulk_name_to_idx(self.parameters["fliC_id"], bulk_ids)
            self.fliD_idx = bulk_name_to_idx(self.parameters["fliD_id"], bulk_ids)
            self.flagellum_idx = bulk_name_to_idx(
                self.parameters["flagellum_id"], bulk_ids
            )
            self.fliS_idx = bulk_name_to_idx(self.parameters["fliS_id"], bulk_ids)
            self.flis_flic_cplx_idx = bulk_name_to_idx(
                self.parameters["flis_flic_cplx_id"], bulk_ids
            )

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        nascent = states["nascent_flagellum"]
        (lengths,) = attrs(nascent, ["filament_length"])
        n_active = lengths.size
        if n_active == 0:
            return next_update

        dt = states["timestep"]
        desired = np.round(self.rate_a / (self.rate_b + lengths) * dt).astype(np.int64)
        desired = np.maximum(desired, 0)

        # Old (pre-FliS-recycling) version, free FliC only, kept per
        # standing preserve-old-code rule:
        # fliC_available = counts(states["bulk"], self.fliC_idx)
        # total_desired = int(desired.sum())
        # if total_desired > fliC_available and total_desired > 0:
        #     # Fair-share: multiple simultaneous filaments compete for the
        #     # same free FliC pool -- scale everyone down proportionally
        #     # rather than letting draw order create bias.
        #     scale = fliC_available / total_desired
        #     desired = np.floor(desired * scale).astype(np.int64)

        # FliS chaperone recycling (2026-08-21, see module docstring): the
        # real, protected supply is FLIS-FLIC-CPLX + free FliC combined --
        # fair-share against the COMBINED pool, not free FliC alone.
        fliC_available_free = counts(states["bulk"], self.fliC_idx)
        fliC_available_protected = counts(states["bulk"], self.flis_flic_cplx_idx)

        # NUMERICAL-STABILITY CAP (added 2026-08-21, NOT a biological
        # parameter): draining FLIS-FLIC-CPLX in one large single-tick jump
        # occasionally destabilizes ecoli-equilibrium's legacy ODE solver
        # for this reaction specifically -- FliS:FliC binding is tight
        # (Kd=5.26e-8 M), and a large abrupt perturbation to the complex/
        # free-FliS/free-FliC triple can push the requested forward extent
        # past what's available, overshooting to negative counts inside the
        # solver (confirmed directly via diagnostic instrumentation: a
        # 12,000s test run failed with the solver requesting a forward
        # extent of ~2,375 against only ~446 free FliS / ~1,926 free FliC
        # available -- both hit negative by the identical delta, confirming
        # it's this one reaction overshooting). This cap spreads the
        # complex's release out gradually across multiple ticks instead of
        # draining it in one shot, which is also more mechanistically
        # honest anyway (real chaperone turnover is continuous, not an
        # instant full-pool drain). Applied BEFORE the fair-share scaling
        # below so total filament growth is correctly reduced when the cap
        # binds, rather than overdrawing free FliC to compensate.
        # STALE as of 2026-08-28: the legacy ODE solver this cap guarded
        # against no longer handles FLIS-FLIC-CPLX_RXN at all -- that
        # reaction now has its own dedicated exact closed-form Step
        # (flagella_flis_flic_equilibrium.py), which cannot overshoot
        # negative by construction (see that Step's module docstring).
        # The 0.3 figure was never a biological number; it was tuned
        # around a crash mode that no longer exists. Investigating
        # (2026-08-28) whether removing it changes the sharp
        # crash-and-recover shape seen in FLIS-FLIC-CPLX population
        # dynamics, driven by this draw racing the equilibrium Step's own
        # every-tick re-solve. Old line kept per standing preserve-old-
        # code rule:
        # MAX_COMPLEX_DRAW_FRACTION = 0.3
        # complex_draw_cap = int(fliC_available_protected * MAX_COMPLEX_DRAW_FRACTION)
        complex_draw_cap = int(fliC_available_protected)

        fliC_available_total = fliC_available_free + complex_draw_cap
        total_desired = int(desired.sum())
        if total_desired > fliC_available_total and total_desired > 0:
            # Fair-share: multiple simultaneous filaments compete for the
            # same combined FliC pool -- scale everyone down proportionally
            # rather than letting draw order create bias.
            scale = fliC_available_total / total_desired
            desired = np.floor(desired * scale).astype(np.int64)

        new_lengths = lengths + desired
        did_complete = new_lengths >= self.target_length
        n_complete = int(did_complete.sum())

        fliC_consumed = int(desired.sum())
        (protein_mass,) = attrs(nascent, ["massDiff_protein"])
        fliC_subunit_mass = states["bulk"]["protein_submass"][self.fliC_idx]
        new_protein_mass = protein_mass + desired * fliC_subunit_mass

        # Draw preferentially from the protected (FliS-escorted) pool
        # first, up to the numerical-stability cap -- exporting/
        # incorporating that FliC releases its chaperone back to the free
        # FliS pool, matching the real escort-cycle mechanism. Only fall
        # back to free (unprotected) FliC for any remaining demand.
        from_complex = min(fliC_consumed, complex_draw_cap)
        from_free = fliC_consumed - from_complex

        bulk_updates = []
        if from_complex > 0:
            bulk_updates.append((self.flis_flic_cplx_idx, -from_complex))
            bulk_updates.append((self.fliS_idx, from_complex))
        if from_free > 0:
            bulk_updates.append((self.fliC_idx, -from_free))
        if n_complete > 0:
            bulk_updates.append(
                (self.fliD_idx, -n_complete * self.fliD_per_completion)
            )
            bulk_updates.append((self.flagellum_idx, n_complete))
        if bulk_updates:
            next_update["bulk"] = bulk_updates

        next_update["nascent_flagellum"] = {
            "delete": np.where(did_complete)[0],
            "set": {
                "filament_length": new_lengths,
                "massDiff_protein": new_protein_mass,
            },
        }
        return next_update
