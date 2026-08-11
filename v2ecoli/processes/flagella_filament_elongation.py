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
dL/dt = a / (b + L), with these reported numbers used to derive the
subunit-count version below:
  - initial growth rate ~83-100 nm/min
  - ~2130 flagellin subunits per micron of filament (paper's own conversion)
  - characteristic length b ~= 0.27 um

Converting to subunit counts (this Step's natural unit, since
filament_length is tracked as an integer subunit count):
  initial_rate = (83 to 100 nm/min) * (2130 subunits/um) / 60s
               ~= 42-50 subunits/s at L=0
  b_subunits = 0.27 um * 2130 subunits/um ~= 575 subunits
  a = rate(0) * b ~= 46 (midpoint) * 575 ~= 26,450 subunit^2/s

  rate(L) = 26450 / (575 + L)  [subunits/s]

Cross-checked against a SEPARATE number in the same paper (filaments
reaching ~10 um, ~21,300 subunits, over 180+ minutes in real-time imaging):
implies an average rate of ~2 subunits/s across the whole growth process,
consistent with a model starting at ~46/s and decaying toward ~1/s near
completion. This is a simplified phenomenological fit to the paper's own
reported summary numbers, not a full re-implementation of their diffusion
PDE -- flagged plainly, same standard as other estimates this session.

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

Ordered in the composite flow: after ecoli-flagella-filament-nucleation.
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
        # converted to subunit-count units -- see module docstring.
        "rate_a": {"_type": "float", "_default": 26450.0},
        "rate_b": {"_type": "float", "_default": 575.0},
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

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        nascent = states["nascent_flagellum"]
        (lengths,) = attrs(nascent, ["filament_length"])
        n_active = lengths.size
        if n_active == 0:
            return next_update

        dt = states["timestep"]
        desired = np.round(self.rate_a / (self.rate_b + lengths) * dt).astype(np.int64)
        desired = np.maximum(desired, 0)

        fliC_available = counts(states["bulk"], self.fliC_idx)
        total_desired = int(desired.sum())
        if total_desired > fliC_available and total_desired > 0:
            # Fair-share: multiple simultaneous filaments compete for the
            # same free FliC pool -- scale everyone down proportionally
            # rather than letting draw order create bias.
            scale = fliC_available / total_desired
            desired = np.floor(desired * scale).astype(np.int64)

        new_lengths = lengths + desired
        did_complete = new_lengths >= self.target_length
        n_complete = int(did_complete.sum())

        fliC_consumed = int(desired.sum())
        (protein_mass,) = attrs(nascent, ["massDiff_protein"])
        fliC_subunit_mass = states["bulk"]["protein_submass"][self.fliC_idx]
        new_protein_mass = protein_mass + desired * fliC_subunit_mass

        bulk_updates = []
        if fliC_consumed > 0:
            bulk_updates.append((self.fliC_idx, -fliC_consumed))
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
