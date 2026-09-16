"""FlgM:FliA equilibrium binding -- exact closed-form solve, real Step.

Added 2026-09-01, Maya Abdalla's flagella-cascade investigation.

Why separate from the shared ecoli-equilibrium Step
-----------------------------------------------------
FLGM-FLIA-CPLX_RXN lived in the shared ~150-reaction equilibrium system
(equilibrium.py), solved every tick by scipy solve_ivp -- the same
solver that crashed repeatedly on FLIS-FLIC-CPLX_RXN (see
flagella_flis_flic_equilibrium.py) for the same reason: its default
atol (1e-6 M) dwarfs a real molecule's concentration here (~1.6e-9 M).
FlgM:FliA's real Kd (~1.8e-10 M, Chadsey et al. 1998) is ~290x TIGHTER
than FliS:FliC's (5.26e-8 M) -- worse for that solver, not better.

The old workaround weakened the model's Kd to 2e-7 M (~1000x weaker
than real) purely to keep the shared solver's answer away from zero --
a stability patch, not biology. Confirmed 2026-09-01: the crash was
always about the SOLVER, not Kd tightness -- the exact closed-form
solve here can't overshoot negative at any Kd, same proof as FliS:FliC.
That removes the reason Kd was ever weakened, so this Step uses the
real value.

This Step replaces FLGM-FLIA-CPLX_RXN's role entirely; the shared
system's copy is zeroed (not deleted -- see flagella_flis_flic_
equilibrium.py's docstring for why) so the other ~150 reactions there
are untouched.

The math
--------
Simple 1:1:1 binding (FlgM + FliA <-> FLGM-FLIA-CPLX -- Chadsey et al.
1998's SPR reports a direct 1:1 Kd, no oligomeric complication like
FliS:FliC's dimer correction). A = free FlgM, B = free FliA, C =
complex; conserved A_tot = A+C, B_tot = B+C (flgm_secretion.py drains
free FlgM only, never touches C or FliA, so both totals hold exactly
within one firing). At equilibrium:

    Kd = A*B/C = (A_tot-C)(B_tot-C)/C

Same quadratic form as flagella_flis_flic_equilibrium.py (see that file
for the full derivation):

    C^2 - (A_tot+B_tot+Kd)*C + A_tot*B_tot = 0
    C = [(A_tot+B_tot+Kd) - sqrt((A_tot+B_tot+Kd)^2 - 4*A_tot*B_tot)] / 2

(smaller root is physical). One closed-form solve per firing -- no
iteration, no tolerance, no overshoot possible.

Kd: 1.8e-10 M, Chadsey, Karlinsey & Hughes 1998, Genes Dev 12:3123
(Salmonella SPR: ka=8.9e5/M/s, kd=1.6e-4/s -> kd/ka=1.8e-10 M,
consistent with the paper's own separately-reported Kd~2e-10 M). Real
value, no longer relaxed. Molar->count conversion via cell volume, same
as the shared Step and flagella_flis_flic_equilibrium.py.

Relaxation-timescale check (2026-09-01): the isolated dissociation
half-life (ln(2)/kd = ~72 min) is the WRONG number to judge against --
it ignores rebinding. Accounting for both directions,
tau = 1/(ka*(A_eq+B_eq)+kd) ~= 1-3s at this cell's real FlgM/FliA scale
-- comparable to the 2s tick, not 72 min. Solving to exact equilibrium
every firing is reasonable on that basis.

Ordered right after ecoli-flagella-flgm-secretion (drains free FlgM),
so this Step's re-solve reflects that tick's own fresh FlgM level --
matching secretion's docstring ("as FlgM drops, equilibrium shifts
toward releasing free FliA") as same-tick causation, vs. the shared
Step's old position (well before secretion). Confirmed real but
unlikely to matter: FlgM changes little per 2s tick, so a one-tick lag
barely shifts the answer either way -- same-tick placement matches the
biology's description, not because the difference is expected to be
visible.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units


NAME = "ecoli-flagella-flgm-flia-equilibrium"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flgm_flia_equilibrium"),
    "global_time": ("global_time",),
}


class FlagellaFlgMFliAEquilibrium(Step):
    """Exact closed-form FlgM:FliA equilibrium -- see module docstring."""

    description = (
        "FlagellaFlgMFliAEquilibrium — exact 1:1:1 binding equilibrium, no ODE.\n\n"
        "    A_tot = free_FlgM + FLGM-FLIA-CPLX\n"
        "    B_tot = free_FliA + FLGM-FLIA-CPLX\n"
        "    C = [(A_tot+B_tot+Kd) - sqrt((A_tot+B_tot+Kd)^2 - 4*A_tot*B_tot)] / 2\n"
        "  Sets FLGM-FLIA-CPLX to the true equilibrium point directly, every firing,\n"
        "  using the real Kd (1.8e-10 M, Chadsey et al. 1998) instead of the shared\n"
        "  solver's deliberately-relaxed 2e-7 M."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flgM_id": {"_type": "string", "_default": "G369-MONOMER[c]"},
        "fliA_id": {"_type": "string", "_default": "EG11355-MONOMER[c]"},
        "flgm_flia_cplx_id": {"_type": "string", "_default": "FLGM-FLIA-CPLX[c]"},
        # Real, cited value (Chadsey, Karlinsey & Hughes 1998, Genes Dev
        # 12:3123, SPR, Salmonella, kd/ka=1.8e-10 M) -- the shared
        # equilibrium system used a deliberately relaxed 2e-7 M instead,
        # purely to avoid that solver's crash mode. This Step's exact
        # solve can't overshoot regardless of Kd, so the real value is
        # used directly. See module docstring.
        "kd_molar": {"_type": "float", "_default": 1.8e-10},
        "cell_density": {"_type": "float", "_default": 1100.0},  # g/L, same
        # real constant used throughout this codebase.
        "n_avogadro": {"_type": "float", "_default": 6.02214076e23},  # /mol
        "timestep": {"_type": "float", "_default": 2.0},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "quantity[float,fg]", "_default": 0},
                },
            },
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
        self.kd_molar = self.parameters["kd_molar"]
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.flgM_idx = None
        self.fliA_idx = None
        self.cplx_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flgM_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flgM_idx = bulk_name_to_idx(self.parameters["flgM_id"], bulk_ids)
            self.fliA_idx = bulk_name_to_idx(self.parameters["fliA_id"], bulk_ids)
            self.cplx_idx = bulk_name_to_idx(
                self.parameters["flgm_flia_cplx_id"], bulk_ids)

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        free_flgM = int(counts(states["bulk"], self.flgM_idx))
        free_fliA = int(counts(states["bulk"], self.fliA_idx))
        cplx = int(counts(states["bulk"], self.cplx_idx))

        a_tot = free_flgM + cplx
        b_tot = free_fliA + cplx
        if a_tot == 0 or b_tot == 0:
            # Nothing to bind at all -- no-op, matches the physical answer
            # (C=0) exactly, no need to run the quadratic.
            return next_update

        # cell_volume = cell_mass / cell_density  [g / (g/L) = L], same
        # conversion the shared equilibrium Step and
        # flagella_flis_flic_equilibrium.py both already use.
        cell_mass_g = (
            as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        ).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density

        # Kd in molecule-count terms, for THIS cell's current real volume.
        kd_counts = self.kd_molar * self.n_avogadro * cell_volume_L

        s = a_tot + b_tot + kd_counts
        discriminant = s * s - 4.0 * a_tot * b_tot
        # Guard against tiny negative values from floating-point roundoff
        # right at the a_tot==b_tot edge case -- the true discriminant is
        # never negative for real, non-negative a_tot/b_tot/kd_counts.
        discriminant = max(discriminant, 0.0)
        c_star = (s - np.sqrt(discriminant)) / 2.0
        # c_star is mathematically guaranteed <= min(a_tot, b_tot); clip
        # only to guard against floating-point overshoot at that boundary.
        c_star = min(max(c_star, 0.0), a_tot, b_tot)

        delta = int(round(c_star)) - cplx
        if delta == 0:
            return next_update

        # delta > 0: more complex forms, consuming free FlgM and free
        # FliA. delta < 0: complex dissociates, releasing both back.
        # Exact and mass-conserving by construction -- the same delta
        # applied to all three, opposite signs.
        next_update["bulk"] = [
            (self.flgM_idx, -delta),
            (self.fliA_idx, -delta),
            (self.cplx_idx, delta),
        ]
        return next_update
