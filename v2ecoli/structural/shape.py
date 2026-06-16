"""Shape step — capsule cell geometry from mass.

A process-bigraph ``Step`` implementing the **Shape** process of Skalnik et al.
2023, *Whole-cell modeling of E. coli colonies* (PLoS Comput. Biol.; SM §3.1).
The cell is a **capsule** (cylinder capped by hemispheres) of fixed width ``w``
and fixed density; it grows purely by elongation, so **length derives from
volume** (= mass / density):

    v   = mass / density                                  (density = 1.1 g/mL)
    l   = (v − 4/3·π(w/2)³) / (π(w/2)²) + w               (cylinder + 2 caps)
    a_o = 4π(w/2)² + 2π(w/2)(l − w)                       (outer membrane area)
    a_i = a_o · (1 − f_p)^(2/3)                           (inner membrane area)
    v_p = v · f_p ,   v_c = v · (1 − f_p)                 (periplasm / cytoplasm)

It also returns the pbg-parsimony ``Capsule`` (in Å), so the parsimony 3D build
can use the simulated shape directly: the cell envelope grows with mass, and the
pre-division (max-mass) state is the elongated, about-to-divide cell.
"""
from __future__ import annotations

import math

from process_bigraph import Step
from pbg_parsimony import Capsule

# Unit bridge: 1 g/mL == 1000 fg/fL  (fg = 1e-15 g, fL = 1e-15 L); fL ≈ µm³.
_G_PER_ML_TO_FG_PER_FL = 1000.0


def shape_from_mass(mass_fg: float, width_um: float = 1.0,
                    density_g_per_ml: float = 1.1,
                    periplasm_fraction: float = 0.2) -> dict:
    """Capsule cell shape for a cell of ``mass_fg`` (Skalnik et al. 2023 §3.1).

    Width, density and periplasm fraction are fixed; everything else derives from
    volume = mass/density. Lengths/areas/volumes in µm / µm² / fL(µm³); the
    pbg-parsimony ``Capsule`` (under ``capsule``) is in Å for the packer.
    """
    v = max(0.0, float(mass_fg) / (density_g_per_ml * _G_PER_ML_TO_FG_PER_FL))  # fL = µm³
    w = float(width_um)
    r = w / 2.0                                                    # cap radius, µm
    lcyl = max(0.0, (v - (4.0 / 3.0) * math.pi * r ** 3) / (math.pi * r ** 2))  # µm
    length = lcyl + w                                             # tip-to-tip, µm
    outer_sa = 4.0 * math.pi * r ** 2 + 2.0 * math.pi * r * (length - w)  # µm²
    inner_sa = outer_sa * (1.0 - periplasm_fraction) ** (2.0 / 3.0)
    cap = Capsule(half_len=(lcyl / 2.0) * 1e4, radius=r * 1e4)    # µm → Å
    return {
        "mass_fg": float(mass_fg),
        "density_g_per_ml": density_g_per_ml,
        "width_um": w,
        "volume_fl": v,
        "length_um": length,
        "outer_sa_um2": outer_sa,
        "inner_sa_um2": inner_sa,
        "periplasm_vol_fl": v * periplasm_fraction,
        "cytoplasm_vol_fl": v * (1.0 - periplasm_fraction),
        "radius_A": cap.radius,
        "half_len_A": cap.half_len,
        "capsule": cap,
    }


# Backwards-compatible alias (the step originally exposed only the capsule).
capsule_from_mass = shape_from_mass


class ShapeStep(Step):
    """Compute capsule cell shape from mass (fixed width, density, periplasm frac)."""

    config_schema = {
        "width_um": {"_type": "float", "_default": 1.0},          # fixed cell width (diameter)
        "density_g_per_ml": {"_type": "float", "_default": 1.1},   # fixed cell density
        "periplasm_fraction": {"_type": "float", "_default": 0.2}, # f_p
    }

    def inputs(self):
        return {"mass_fg": "float"}

    def outputs(self):
        return {"shape": "any"}

    def update(self, state, interval=None):
        mass = float(state.get("mass_fg") or 0.0)
        return {"shape": shape_from_mass(
            mass, self.config["width_um"], self.config["density_g_per_ml"],
            self.config["periplasm_fraction"])}
