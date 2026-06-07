"""
=========================
DnaA-box Binding Listener  (dnaa-3)
=========================

In-sim OBSERVABLE of DnaA-box occupancy. Each tick it reads the free DnaA-ATP /
DnaA-ADP bulk counts + cell mass (→ volume → concentration), and computes the
fast-equilibrium occupancy P = C/(C+K_d) for each affinity pool, emitting the
``dnaA_binding`` listener (the interface the box-occupancy viz expects).

This is the dnaa-3 binding model as a READ-ONLY listener: it does NOT mutate
bulk counts or the DnaA_box flags, so the validated dnaa-2 cell cycle is
preserved exactly. Making box binding a real SINK (partitioning the bulk DnaA
pool) is the next, separately-validated increment.

Pools (dnaa-3 box catalog): chromosomal_high (~302 consensus, K_d≈1 nM, ATP+ADP),
oriC_high (3 R1/R2/R4, K_d≈1 nM, ATP+ADP), oriC_low (8, K_d≈100 nM, ATP only),
promoter_high (2, K_d≈1 nM, ATP+ADP). The consensus boxes are classified live
from the DnaA_box unique store by region; the 8 oriC-low are a fixed design pool.

WIP (2026-06-05): the binding model + I/O is complete and the step instantiates
cleanly (params + topology verified), but wiring it into baseline.py hits TWO
distinct, characterized blockers — NOT yet resolved, so it is NOT registered:

  1. As a LAYER-7 listener (alongside ecoli-mass-listener): the engine never
     invokes ``update`` (no firing). replication_data — same DnaA_box topology,
     same layer — fires every tick; the difference is this step's bulk +
     cross-listener ``listeners.mass`` read. Reading a store written by another
     step in the SAME parallel layer appears to block scheduling. (equilibrium
     reads listeners.mass fine, but from an EARLIER layer = prior-tick value.)
  2. In an EARLY layer (next to ecoli-equilibrium, so mass reads the prior tick
     like equilibrium does): adding the step changes the composite ``inputs_hash``
     → the dnaa-2 cache (out/cache_dnaa2) is rejected as stale before the sim
     even builds. Fixing needs a full ParCa + cache REBUILD for the new composite.

Next session: rebuild ParCa+cache for the composite WITH this step in an early
layer, then confirm firing + occupancy. Until then the dnaa-3 occupancy figure
uses the equivalent fast-equilibrium computation post-hoc
(scripts/render_dnaa3_occupancy.py) on the validated dnaa-2 baseline — same model,
same numbers.
"""
import numpy as np
from v2ecoli.library.schema import bulk_name_to_idx, counts, attrs
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity

NAME = "dnaa_box_binding_listener"
TOPOLOGY = {
    "bulk": ("bulk",),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "mass_listener": ("listeners", "mass"),   # READ port (separate from the write port)
    "listeners": ("listeners",),               # WRITE port (dnaA_binding output)
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}

CHROM_LEN = 4_641_652  # bp; DnaA_box coords are oriC-relative signed
# DnaA bulk species (apo / DnaA-ATP / DnaA-ADP) — same IDs dnaa-2 uses.
APO, ATP, ADP = "PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"


def _region(signed):
    if abs(int(signed)) <= 5_000:
        return "oriC"
    if -50_000 < int(signed) < -30_000:
        return "dnaA_promoter"
    return "chromosomal"


class DnaaBoxBinding(Step):
    """Read-only DnaA-box occupancy listener (fast-equilibrium model)."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "time_step": "float{1.0}",
        "cell_density": "float{1100.0}",   # g/L (≈1.1 g/mL)
        "n_avogadro": "float{6.02214076e23}",
        # per-pool K_d (nM) + sizes; the 8 oriC-low boxes are a fixed design pool
        "kd_high_nM": "float{1.0}",
        "kd_low_nM": "float{100.0}",
        "n_oric_low": "integer{8}",
        # SINK MODE (dnaa-3 Rashmi item 1). When False (default) the step is a
        # pure read-only OBSERVER: it never returns a ``bulk`` delta, so the
        # validated dnaa-2 cell cycle is preserved exactly. When True it becomes
        # a real SINK: the DnaA molecules it computes as bound are SUBTRACTED
        # from the free bulk pool (partition free vs bound), which lowers free
        # DnaA-ATP and the t=0 low-affinity occupancy. The sink MUTATES bulk and
        # therefore perturbs the cell cycle — it needs separate re-validation.
        "sink": "boolean{false}",
    }

    def initialize(self, config):
        # EcoliStep calls initialize() after building self.parameters (NOT __init__).
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.kd_high = self.parameters["kd_high_nM"]
        self.kd_low = self.parameters["kd_low_nM"]
        self.n_oric_low = self.parameters["n_oric_low"]
        self.sink = bool(self.parameters.get("sink", False))
        self.molecule_idx = None

    def inputs(self):
        return {
            # Use the registered ``bulk_array`` type (NOT numpy_schema("bulk"))
            # so the bulk store keeps its BulkNumpyUpdate apply. numpy_schema
            # returns a typeless dict whose _updater function is dropped on
            # schema merge — the store then falls back to the generic Array
            # apply (state[idx] += delta), which crashes when the NEXT step
            # (Equilibrium) returns a [(idx, delta)] bulk delta against the
            # structured bulk dtype. This is the same convention counts_deriver
            # and mass_deriver use for their read-only bulk port.
            "bulk": {"_type": "bulk_array", "_default": []},
            "DnaA_boxes": {"_type": DNAA_BOX_ARRAY, "_default": []},
            "mass_listener": {"cell_mass": {"_type": "quantity[float,fg]", "_default": 0}},
            "global_time": {"_type": "float", "_default": 0.0},
            "timestep": {"_type": "float", "_default": 1},
        }

    def outputs(self):
        out = {}
        # SINK mode adds a writable bulk delta port (bulk_array updater format:
        # [(idx_array, delta_array)]). In OBSERVER mode bulk is read-only and
        # absent from outputs(), so no bulk delta is ever produced.
        if getattr(self, "sink", False):
            out["bulk"] = "bulk_array"
        out["listeners"] = {
                "dnaA_binding": {
                    "free_DnaA_ATP_nM": {"_type": "overwrite[float]", "_default": []},
                    "free_DnaA_ADP_nM": {"_type": "overwrite[float]", "_default": []},
                    "oric": {
                        "high_affinity_occupied": {"_type": "overwrite[float]", "_default": []},
                        "low_affinity_occupied": {"_type": "overwrite[float]", "_default": []},
                        "n_bound": {"_type": "overwrite[integer]", "_default": []},
                        "n_total": {"_type": "overwrite[integer]", "_default": []},
                    },
                    "dnaap": {
                        "occupied": {"_type": "overwrite[float]", "_default": []},
                        "n_bound": {"_type": "overwrite[integer]", "_default": []},
                        "n_total": {"_type": "overwrite[integer]", "_default": []},
                    },
                    "chromosome": {
                        "occupied": {"_type": "overwrite[float]", "_default": []},
                        "occupied_count": {"_type": "overwrite[integer]", "_default": []},
                        "n_total": {"_type": "overwrite[integer]", "_default": []},
                    },
                    "total_DnaA_bound": {"_type": "overwrite[integer]", "_default": []},
                }
        }
        return out

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx([APO, ATP, ADP], states["bulk"]["id"])
        apo, atp, adp = counts(states["bulk"], self.molecule_idx)

        cell_mass_g = as_quantity(states["mass_listener"]["cell_mass"], units.fg).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density if cell_mass_g > 0 else None

        def nM(n):
            if not cell_volume_L:
                return 0.0
            return float(n) / (cell_volume_L * self.n_avogadro) * 1e9

        atp_nM, adp_nM = nM(atp), nM(adp)
        c_high = atp_nM + adp_nM            # high-aff pools bind ATP or ADP
        c_low = atp_nM                       # oriC-low binds ATP only
        p_high = c_high / (c_high + self.kd_high) if c_high else 0.0
        p_low = c_low / (c_low + self.kd_low) if c_low else 0.0

        # classify the live consensus boxes by region
        (coords,) = attrs(states["DnaA_boxes"], ["coordinates"])
        n = {"oriC": 0, "dnaA_promoter": 0, "chromosomal": 0}
        for c in coords:
            n[_region(c)] += 1
        oric_high_n, dnaap_n, chrom_n = n["oriC"], n["dnaA_promoter"], n["chromosomal"]
        oric_low_n = self.n_oric_low

        oric_bound = int(round(oric_high_n * p_high + oric_low_n * p_low))
        dnaap_bound = int(round(dnaap_n * p_high))
        chrom_bound = int(round(chrom_n * p_high))
        total_bound = oric_bound + dnaap_bound + chrom_bound

        update = {"listeners": {"dnaA_binding": {
            "free_DnaA_ATP_nM": atp_nM, "free_DnaA_ADP_nM": adp_nM,
            "oric": {"high_affinity_occupied": p_high, "low_affinity_occupied": p_low,
                     "n_bound": oric_bound, "n_total": oric_high_n + oric_low_n},
            "dnaap": {"occupied": p_high, "n_bound": dnaap_bound, "n_total": dnaap_n},
            "chromosome": {"occupied": p_high, "occupied_count": chrom_bound, "n_total": chrom_n},
            "total_DnaA_bound": total_bound,
        }}}

        # SINK MODE: partition the free DnaA pool. The molecules we just
        # computed as bound at boxes are REMOVED from the free bulk pool so they
        # no longer count toward free DnaA-ATP. We sink from the ATP-bound form
        # (the active initiator) — all box pools bind ATP-DnaA (the high-affinity
        # pools also accept ADP, but partitioning the active ATP form is the
        # physically meaningful lever for lowering free DnaA-ATP and t=0
        # low-affinity occupancy). Never sink more than the available free pool.
        # This is the ONLY path that returns a ``bulk`` delta; in observer mode
        # the step stays read-only.
        if self.sink and total_bound > 0:
            atp_count = int(atp)
            sink_n = min(total_bound, atp_count)
            if sink_n > 0:
                # bulk_array updater format: list of (indices, deltas). ATP is
                # the 2nd entry of self.molecule_idx ([APO, ATP, ADP]).
                atp_idx = int(np.asarray(self.molecule_idx)[1])
                update["bulk"] = [(np.array([atp_idx]), np.array([-sink_n]))]

        return update
