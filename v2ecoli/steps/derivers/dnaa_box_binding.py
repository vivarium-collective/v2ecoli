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
import os
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
        # CHROMOSOMAL HIGH-AFFINITY CAPACITY (dnaa-3 Rashmi item 1, sweepable).
        # The DnaA_box unique store carries ~302 loose-consensus (TTWTNCACA)
        # chromosomal boxes. ChIP-seq (Smith 2024) finds only ~13 STRONGLY
        # DnaA-bound regions / 32 strict-consensus sites genome-wide — DnaA
        # prefers closely-spaced box PAIRS bound by dimers, so the ~302
        # loose-motif pool is NOT the in-vivo titration sink. This cap limits the
        # chromosomal high-affinity pool that the SINK titrates (and reports as
        # chromosome.occupied_count) to the strongly-bound count. Default 32 =
        # the ChIP-seq strong-site count (Smith 2024: ~13-32 strongly-bound
        # regions; the ~302 loose-consensus pool is NOT the in-vivo titration).
        # Chosen from the full-trajectory capacity sweep (55-min observer
        # artifacts, 3301 ticks, cache_dnaa3 seed 1; caps 302/32/13 -> sink free
        # DnaA-ATP mean 0.4 / 44.2 / 128.7 nM): 302 over-titrates free DnaA to
        # ~0; 13 leaves free DnaA ~129 nM (back ABOVE the 100 nM oriC-low K_d,
        # oriC-low occ ~0.53 — the over-binding Rashmi flagged); 32 lands free
        # DnaA-ATP ~44 nM (median 14, out of the >K_d over-binding regime) with
        # LOW early-cycle oriC low-aff occupancy (~0.05, resolving the read-only
        # listener's t=0 ~0.52) rising to ~0.51 late-cycle toward initiation, and
        # the gen-1 cell cycle preserved (divides at 60 min, oriC reaches 2, zero
        # re-init). -1 = ALL live boxes (~302, legacy / read-only listener).
        # Overridable per-run via env var DNAA_N_CHROM_HIGH_CAP.
        "n_chrom_high_cap": "integer{-1}",
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
        # Chromosomal high-affinity capacity cap (sweepable). Env var wins over
        # the config default so a sweep runner can set it per variant without
        # rebuilding the composite. -1 means "no cap" (use all live boxes).
        cap = self.parameters.get("n_chrom_high_cap", -1)
        env_cap = os.environ.get("DNAA_N_CHROM_HIGH_CAP")
        if env_cap is not None and env_cap != "":
            cap = int(env_cap)
        self.n_chrom_high_cap = int(cap)
        self.molecule_idx = None
        # Sink bookkeeping: how many DnaA-ATP molecules this step has ALREADY
        # removed from the free pool (currently held bound). Each tick we move
        # only the NET change in bound count (Δbound) — binding removes from
        # free, release returns to free — instead of re-subtracting the whole
        # bound amount every tick (which would drain the pool to zero).
        self._sunk_atp = 0

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

        # Effective chromosomal high-affinity capacity. The unique store carries
        # ~302 loose-consensus boxes, but ChIP-seq finds only ~13-32 strongly
        # DnaA-bound sites genome-wide. When a cap is set (>=0) the chromosomal
        # pool that titrates DnaA (occupied_count + the sink) is limited to that
        # strongly-bound count; the box catalog (n_total) still reports the true
        # ~302 so the doubling-at-replication readout is unchanged. The cap
        # scales WITH replication (it tracks chrom_n's fraction) so a 2x box
        # count → 2x capped sites, preserving the per-genome capacity.
        chrom_n_eff = chrom_n
        if self.n_chrom_high_cap >= 0 and chrom_n > 0:
            # Scale the cap by the per-genome box multiplicity (chrom_n doubles
            # at replication ~302->604) so the cap stays per-genome.
            mult = max(1.0, chrom_n / 302.0)
            chrom_n_eff = min(chrom_n, int(round(self.n_chrom_high_cap * mult)))

        oric_bound = int(round(oric_high_n * p_high + oric_low_n * p_low))
        dnaap_bound = int(round(dnaap_n * p_high))
        chrom_bound = int(round(chrom_n_eff * p_high))
        total_bound = oric_bound + dnaap_bound + chrom_bound

        update = {"listeners": {"dnaA_binding": {
            "free_DnaA_ATP_nM": atp_nM, "free_DnaA_ADP_nM": adp_nM,
            "oric": {"high_affinity_occupied": p_high, "low_affinity_occupied": p_low,
                     "n_bound": oric_bound, "n_total": oric_high_n + oric_low_n},
            "dnaap": {"occupied": p_high, "n_bound": dnaap_bound, "n_total": dnaap_n},
            "chromosome": {"occupied": p_high, "occupied_count": chrom_bound, "n_total": chrom_n_eff},
            "total_DnaA_bound": total_bound,
        }}}

        # SINK MODE: partition the free DnaA pool. The molecules computed as
        # bound at boxes are held OUT of the free bulk pool so they no longer
        # count toward free DnaA-ATP. Binding is an equilibrium, so we move only
        # the NET change vs what we already hold sunk (Δ = desired_bound −
        # already_sunk): a rise in occupancy removes more from free, a fall
        # RETURNS DnaA to free. (Subtracting the full bound amount every tick —
        # the naive first cut — drained the pool to ~0 nM and starved
        # initiation.) We sink the ATP-bound form (the active initiator); all box
        # pools bind ATP-DnaA. Clamp so we never drive free ATP negative and
        # never hold more sunk than were ever bound. This is the ONLY path that
        # returns a ``bulk`` delta; in observer mode the step stays read-only.
        if self.sink:
            atp_count = int(atp)
            desired_sunk = max(0, total_bound)
            delta = desired_sunk - self._sunk_atp        # >0 remove, <0 return
            if delta > 0:
                delta = min(delta, atp_count)            # don't go negative free
            new_sunk = self._sunk_atp + delta
            move = new_sunk - self._sunk_atp
            if move != 0:
                atp_idx = int(np.asarray(self.molecule_idx)[1])
                # bulk_array updater format: list of (indices, deltas).
                update["bulk"] = [(np.array([atp_idx]), np.array([-move]))]
                self._sunk_atp = new_sunk

        return update
