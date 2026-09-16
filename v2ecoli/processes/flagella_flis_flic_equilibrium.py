# FliS-FliC chaperone binding -- exact closed-form solve.
#
# A = free FliS monomer count, D = free FliS dimer count (=A // 2,
# fast-pre-equilibrium), B = free FliC, C = FLIS-FLIC-CPLX. Conserved:
# D_tot = D + C, B_tot = B + C (only this reaction and elongation's
# chaperone-release consume/release these species -- see
# flagella_filament_elongation.py, updated in lockstep). At equilibrium:
#
#     Kd = D * B / C = (D_tot - C)(B_tot - C) / C
#
# Standard quadratic in C:
#
#     C^2 - (D_tot + B_tot + Kd) * C + D_tot * B_tot = 0
#     C = [(D_tot + B_tot + Kd) - sqrt((D_tot + B_tot + Kd)^2 - 4*D_tot*B_tot)] / 2
#
# (smaller root is physical -- the larger exceeds min(D_tot, B_tot)).
# One closed-form solve per firing, no iteration/ODE/tolerance. Only
# change from 1:1:1: each unit of C moves 2 raw FliS monomers, not 1 --
# mathematically guaranteed not to overdraw free FliS (2*(D_tot-cplx) <= A
# always, since D_tot-cplx = A // 2).
#
# Kd: 5.26e-8 M, Muskotal et al. 2006 (FEBS Lett 580:3916, ITC,
# Ka=1.9e7/M), now understood as dimer:FliC not monomer:FliC. Converted
# molar->molecule-count via cell volume each firing (same conversion the
# shared equilibrium Step uses).


import numpy as np
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units

NAME = "ecoli-flagella-flis-flic-equilibrium"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flis_flic_equilibrium"),
    "global_time": ("global_time",),
}

class FlagellaFliSFliCEquilibrium(Step):
    topology = TOPOLOGY
    name = NAME


    config_schema = {
        "fliS_id": {"_type": "string", "_default": "EG11388-MONOMER[c]"},
        "fliC_id": {"_type": "string", "_default": "EG10321-MONOMER[e]"},
        "flis_fliC_complex_id" : { "_type": "string", "_default": "FLIS-FLIC-CPLX[e]"},
        # Real, cited value (Muskotal et al. 2006, FEBS Lett 580:3916, ITC,
        # Ka=1.9e7/M) -- unchanged from what the shared equilibrium system
        # used before this Step existed. This IS the biological number;
        # nothing about moving this reaction into its own Step changes it.
        "kd_molar": {"_type": "float", "_default": 5.26e-8},
        "cell_density": {"_type": "float", "_default": 1100.0},
        "n_avogadro": {"_type": "float", "_default": 6.02214076e23},
        "timestep": {"_type": "float", "_default": 2.0},
        }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default":[]},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "quantity[float, fg]", "_default":0},
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
        self.fliS_id = None
        self.fliC_id = None
        self.flis_fliC_complex_id = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.fliS_id is None:
            bulk_ids = states["bulk"]["id"]
            self.fliS_id = bulk_name_to_idx(self.parameters["fliS_id"], bulk_ids)
            self.fliC_id = bulk_name_to_idx(self.parameters["fliC_id"], bulk_ids)
            self.flis_fliC_complex_id = bulk_name_to_idx(self.parameters["flis_fliC_complex_id"], bulk_ids)

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        free_fliS = int(counts(states["bulk"], self.fliS_id))
        free_fliC = int(counts(states["bulk"], self.fliC_id))
        cplx = int(counts(states["bulk"], self.flis_fliC_complex_id))

        #How many dimer equivalents of FliS exist in total
        d_tot = free_fliS // 2 + cplx

        #Total FliC -- whether currently free or already in complex
        b_tot = free_fliC + cplx

        #If nothing to bind -- no need to run the quadratic equation
        if d_tot == 0 or b_tot == 0:
            return next_update

        #cell_volume = cell_mass / cell_density [g/ (g/L) = L], same conversion as the equilibirum step
        cell_mass_g = (
            as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density

        #Kd in molecule-count terms, for this cell current real volume
        # kd_counts = 5.26e-8 mol/L * 6.02e23 molecules/mol * L = units in molecules
        # necessary b/c quadratic equation elsewhere in the step and the rest of math works all in raw molecules counts
        # recomputed every firing too -- cell_volume_L isn't fixed, kd counts changes overtime but the read kd (concentration) does not
        kd_counts = self.kd_molar * self.n_avogadro * cell_volume_L

        s = d_tot + b_tot + kd_counts
        discriminant = s *s - 4.0 * d_tot * b_tot

        #guard against negative values so rounding values dont give negative values
        # safety clamp
        discriminant = max(discriminant, 0.0)

        #c_star is actual quadratic formula, only smaller value (minus) is the valid one, larger one would be more complex than whats available
        c_star = (s - np.sqrt(discriminant)) / 2.0
        #another safety clamp for the actual quadratic
        c_star = min(max(c_star, 0.0), d_tot, b_tot)

        delta = int(round(c_star)) - cplx
        if delta == 0:
            return next_update

        # delta > 0 = more complex forms, consuming free FliS (2 FliS per unit) and one FliC unit.
        # delta < 0 = complex dissociates, releasing both back
        # -2 * delta cant overdraw free_FliS because d_tot - cplx = free_fliS // 2 bounds

        next_update["bulk"] = [
            (self.fliS_id, -2 * delta),
            (self.fliC_id, -delta),
            (self.flis_fliC_complex_id, delta),
        ]
        return next_update














