"""
======================
Chromosome Replication
======================

Performs initiation, elongation, and termination of active partial chromosomes
that replicate the chromosome.

Mathematical Model
------------------
Chromosome replication proceeds in three modules per timestep:

**Module 1 -- Initiation**

Replication initiates when the cell mass per origin of replication (oriC)
exceeds a critical threshold:

    M_cell / n_oriC  >=  M_critical(tau)

where M_critical (fg) depends on the growth rate via the doubling time tau.
Upon initiation, two replisomes are assembled per oriC (one per replichore),
each consuming 3 trimer + 1 monomer subunits (if mechanistic_replisome=True).
Two new chromosome domains are created as children of the parent domain.

**Module 2 -- Elongation**

Each replication fork extends along its template strand. The elongation
rate v (nt/s) is drawn stochastically around the basal rate (default 967 nt/s).
dNTP consumption is computed via the ``polymerize`` algorithm:

    sequences = buildSequences(template, fork_position, v * dt)
    result = polymerize(sequences, dNTP_counts, rate_limit)

If one dNTP species is limiting, all four are scaled by the same ratio
to maintain stoichiometric balance. Mass increase per fork:

    delta_mass_DNA = sum(elongated_nt_i * weight_i)   [fg]

Each dNTP polymerized releases one pyrophosphate (PPi).

**Module 3 -- Termination**

A fork terminates when its coordinate reaches the replichore length:

    |coordinate| == L_replichore

When both forks of a domain terminate, the domain splits: the parent
domain's two child domains become independent chromosomes. Replisome
subunits are released back to the bulk pool. A D-period timer is set
on the new full chromosome for cell division scheduling.
"""

import os

import numpy as np

from v2ecoli.library.schema import (
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    listener_schema,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease

# topology_registry removed
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    ACTIVE_REPLISOME_ARRAY,
    ORIC_ARRAY,
    CHROMOSOME_DOMAIN_ARRAY,
    FULL_CHROMOSOME_ARRAY,
)


# Register default topology for this process, associating it with process name
NAME = "ecoli-chromosome-replication"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "oriCs": ("unique", "oriC"),
    "chromosome_domains": ("unique", "chromosome_domain"),
    "full_chromosomes": ("unique", "full_chromosome"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "timestep": ("timestep",),
    "global_time": ("global_time",),
}


class ChromosomeReplication(Step):
    """Chromosome Replication Step

    dNTPs are consumed only by chromosome replication; no other process
    competes for them. Runs as a plain Step.
    """

    description = (
        "Chromosome Replication — initiate, elongate, and terminate replication forks.\n\n"
        "1. Initiation: fire when  M_cell / n_oriC ≥ M_critical(τ);  +2 replisomes·oriC, +2 domains.\n"
        "2. Elongation: seqs = buildSequences(template, pos, ν·dt); polymerize(seqs, dNTPs, limit);\n"
        "   Δm_DNA = ∑ᵢ (elongated_ntᵢ · weightᵢ);  PPi released = dNTP polymerized.\n"
        "3. Termination: fork ends when |coordinate| = L_replichore; domain splits, subunits recycled.\n"
        "  M_cell: cell mass (fg); n_oriC: oriC count; M_critical(τ): mass/oriC threshold at doubling time τ;\n"
        "  ν: stochastic elongation rate (nt/s); dt: timestep; L_replichore: replichore length (nt)."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'D_period': {'_type': 'node', '_default': np.array([], dtype=float)},
        'basal_elongation_rate': {'_type': 'integer[nt/s]', '_default': 967},
        'criticalInitiationMass': {'_type': 'quantity[fg]', '_default': 975.0},
        'dntps': {'_type': 'list[string]', '_default': []},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'get_dna_critical_mass': {'_type': 'method', '_default': None},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'mechanistic_replisome': {'_type': 'boolean', '_default': True},
        'no_child_place_holder': {'_type': 'integer', '_default': -1},
        'nutrientToDoublingTime': {'_type': 'map[quantity[float,min]]', '_default': {}},
        'polymerized_dntp_weights': {'_type': 'quantity[array[float],fg]', '_default': []},
        'ppi': {'_type': 'list[string]', '_default': []},
        'replication_coordinate': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replichore_lengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replisome_monomers_subunits': {'_type': 'list[string]', '_default': []},
        'replisome_protein_mass': {'_type': 'float[fg]', '_default': 0},
        'replisome_trimers_subunits': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'sequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'oriCs': {'_type': ORIC_ARRAY, '_default': []},
            'chromosome_domains': {'_type': CHROMOSOME_DOMAIN_ARRAY, '_default': []},
            'full_chromosomes': {'_type': FULL_CHROMOSOME_ARRAY, '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
                # dnaa-5 mechanistic-initiation diagnostic: read the oriC DnaA-ATP
                # occupancy (written last tick by replication_data_listener) so the
                # initiation trigger can fire on DnaA-ATP filament saturation instead
                # of the cell-mass heuristic. Only USED when
                # DNAA_INITIATION_TRIGGER=mechanistic; harmless (read-only) otherwise.
                'replication_data': {
                    'oriC_high_bound_atp': {'_type': 'integer', '_default': 0},
                    'oriC_low_bound_atp': {'_type': 'integer', '_default': 0},
                    'number_of_oric': {'_type': 'integer', '_default': 0},
                    # per-origin oriC-low saturation for asynchronous initiation
                    'oriC_domain_index': {'_type': 'array[integer]', '_default': []},
                    'oriC_low_bound_atp_by_origin': {'_type': 'array[integer]', '_default': []},
                },
            },
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'global_time': {'_type': 'float', '_default': 0.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'active_replisomes': ACTIVE_REPLISOME_ARRAY,
            'oriCs': ORIC_ARRAY,
            'chromosome_domains': CHROMOSOME_DOMAIN_ARRAY,
            'full_chromosomes': FULL_CHROMOSOME_ARRAY,
            'listeners': {
                'replication_data': {
                    # Critical initiation mass — femtograms
                    'critical_initiation_mass': {'_type': 'overwrite[float[fg]]', '_default': 0.0},
                    # Cell mass / critical mass — dimensionless ratio
                    'critical_mass_per_oriC': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
        }



    def initialize(self, config):

        # Load parameters
        self.get_dna_critical_mass = self.parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = self.parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]

        # dnaa-5 mechanistic-initiation diagnostic (default OFF = baseline mass trigger).
        # DNAA_INITIATION_TRIGGER=mechanistic fires initiation when the oriC high-affinity
        # DnaA boxes are saturated with DnaA-ATP (>= threshold per origin), instead of the
        # cell-mass-per-origin heuristic. DNAA_INIT_HIGH_THRESHOLD sets the per-origin
        # oriC-high ATP count required (default 3 = all 3 high-affinity sites).
        self.initiation_trigger = os.environ.get("DNAA_INITIATION_TRIGGER", "mass").lower()
        self.init_high_threshold = int(os.environ.get("DNAA_INIT_HIGH_THRESHOLD", "3"))
        # dnaa-6 payoff re-test: which oriC pool's DnaA-ATP saturation triggers initiation.
        # 'high' (default) = the first-pass oriC-high (3/3) trigger; 'low' = the oriC-LOW
        # COOPERATIVE switch (the filament-completion signal, dnaa-5) — the biologically
        # correct trigger once cooperativity makes oriC-low switch-like.
        self.init_trigger_pool = os.environ.get("DNAA_INIT_TRIGGER_POOL", "high").lower()
        self.init_low_threshold = int(os.environ.get("DNAA_INIT_LOW_THRESHOLD", "6"))
        # dnaa-7 SeqA eclipse: minutes after an initiation during which the mechanistic
        # trigger is blocked (the post-initiation sequestration that prevents re-firing).
        # Default 0 = no eclipse (baseline / dnaa-6 behavior).
        self.init_eclipse_min = float(os.environ.get("DNAA_INIT_ECLIPSE_MIN", "0"))
        self._last_init_time = -1e18  # time of last initiation (s); -inf so the first fires
        self._cur_time = 0.0          # set each tick in _prepare
        # Asynchronous initiation (Rashmi 2026-07-01, flag-gated, default OFF =
        # synchronous baseline byte-identical). When ON (with the mechanistic
        # oriC-low trigger), each origin fires INDEPENDENTLY when its OWN low-affinity
        # sites saturate (>= init_low_threshold), instead of doubling all origins in
        # one tick. Origins separate via (a) per-origin stochastic box filling and
        # (b) the RIDA feedback: the first origin to fire activates replisome-coupled
        # RIDA, dropping free DnaA-ATP so the others wait — giving oriC 2->3->4, not
        # the biologically-wrong synchronous 2->4 jump.
        self.async_initiation = os.environ.get("DNAA_ASYNC_INITIATION", "0").lower() in ("1", "true", "yes")
        self._last_init_time_by_domain = {}  # per-origin eclipse clock (domain_index -> time s)
        # Dwell time (Rashmi 2026-07-02): an origin must stay SATURATED (>= threshold)
        # continuously for DNAA_INIT_DWELL_SEC seconds before it may fire — so a
        # transient fill-spike does not trigger initiation (the sites must be "stuck").
        # Default 0 = fire immediately on saturation (prior behavior).
        self.init_dwell_sec = float(os.environ.get("DNAA_INIT_DWELL_SEC", "0"))
        self._sat_since_by_domain = {}   # domain_index -> time it first became saturated (s)
        self._sat_since_global = None    # sync path: time the global signal first crossed
        self._mech_ready = False  # set each tick in _request, read in _evolve
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = self.parameters["polymerized_dntp_weights"]
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        # Sim options
        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        # random state
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        # Bulk molecule names
        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters[
            "replisome_monomers_subunits"
        ]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None

    def update(self, states, interval=None):
        self._prepare(states)
        return self._evolve(states)

    def _should_initiate(self, n_oriC):
        """Replication-initiation gate. Default: the cell-mass-per-origin heuristic
        (criticalMassPerOriC >= 1.0). When DNAA_INITIATION_TRIGGER=mechanistic: fire
        on oriC DnaA-ATP saturation (computed in _prepare as _mech_ready), subject to a
        post-initiation ECLIPSE (dnaa-7 SeqA sequestration): for DNAA_INIT_ECLIPSE_MIN
        minutes after an initiation, the trigger is blocked so a just-fired origin cannot
        re-fire (prevents over-initiation). Eclipse=0 (default) disables the block."""
        if self.initiation_trigger == "mechanistic":
            if not self._mech_ready:
                return False
            if self.init_eclipse_min > 0.0:
                if (self._cur_time - self._last_init_time) < self.init_eclipse_min * 60.0:
                    return False  # within the SeqA eclipse — re-initiation blocked
            return True
        return self.criticalMassPerOriC >= 1.0

    def _prepare(self, states):
        timestep = states["timestep"]
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, states["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, states["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, states["bulk"]["id"])
        requests = {}
        # Get total count of existing oriC's
        n_oriC = states["oriCs"]["_entryState"].sum()
        # If there are no origins, return immediately
        if n_oriC == 0:
            return

        # Get current cell mass
        cellMass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)

        # Get critical initiation mass for current simulation environment
        current_media_id = states["environment"]["media_id"]
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id]
        )

        # Calculate mass per origin of replication, and compare to critical
        # initiation mass. If the cell mass has reached this critical mass,
        # the process will initiate a round of chromosome replication for each
        # origin of replication.
        massPerOrigin = cellMass / n_oriC
        # .to('dimensionless') forces pint to reduce mass/mass; otherwise
        # the ratio can retain residual unit scaling and the >= 1.0
        # comparison below wrongly evaluates to False.
        self.criticalMassPerOriC = (
            massPerOrigin / self.criticalInitiationMass
        ).to('dimensionless')

        # dnaa-5 mechanistic-initiation diagnostic: readiness = oriC high-affinity boxes
        # saturated with DnaA-ATP (>= threshold per origin). Computed here (in _request)
        # and read again in _evolve via _should_initiate().
        self._cur_time = float(states.get("global_time", 0.0))  # for the dnaa-7 eclipse clock
        self._fire_domains = None  # set below for async initiation
        if self.initiation_trigger == "mechanistic":
            rd = states["listeners"]["replication_data"]
            if self.init_trigger_pool == "low":
                signal = int(rd.get("oriC_low_bound_atp", 0))   # cooperative filament-completion (dnaa-5)
                threshold = self.init_low_threshold
            else:
                signal = int(rd.get("oriC_high_bound_atp", 0))  # first-pass oriC-high saturation
                threshold = self.init_high_threshold
            saturated = signal >= threshold * n_oriC
            # DWELL: the global signal must stay saturated for init_dwell_sec before firing.
            if self.init_dwell_sec > 0.0:
                if saturated:
                    if self._sat_since_global is None:
                        self._sat_since_global = self._cur_time
                    self._mech_ready = (self._cur_time - self._sat_since_global) >= self.init_dwell_sec
                else:
                    self._sat_since_global = None
                    self._mech_ready = False
            else:
                self._mech_ready = saturated
            # ASYNCHRONOUS initiation (oriC-low trigger only): decide PER ORIGIN which
            # origins are saturated (>= threshold), past their own eclipse, AND have stayed
            # saturated for the dwell time. Only these fire this tick, so origins initiate
            # independently (oriC 2->3->4).
            if self.async_initiation and self.init_trigger_pool == "low":
                dom = np.asarray(rd.get("oriC_domain_index", []), dtype=np.int64)
                per = np.asarray(rd.get("oriC_low_bound_atp_by_origin", []), dtype=np.int64)
                fire = []
                live_domains = set()
                if dom.size and dom.size == per.size:
                    for d, b in zip(dom.tolist(), per.tolist()):
                        live_domains.add(d)
                        if b < threshold:
                            self._sat_since_by_domain.pop(d, None)  # dropped below -> reset dwell
                            continue
                        # per-origin dwell: require sustained saturation
                        if self.init_dwell_sec > 0.0:
                            since = self._sat_since_by_domain.setdefault(d, self._cur_time)
                            if (self._cur_time - since) < self.init_dwell_sec:
                                continue  # saturated, but not long enough yet
                        if self.init_eclipse_min > 0.0:
                            last = self._last_init_time_by_domain.get(d, -1e18)
                            if (self._cur_time - last) < self.init_eclipse_min * 60.0:
                                continue  # this origin is within its own eclipse
                        fire.append(d)
                # forget dwell timers for origins that no longer exist
                for d in list(self._sat_since_by_domain):
                    if d not in live_domains:
                        self._sat_since_by_domain.pop(d, None)
                self._fire_domains = fire
                # gate the whole initiation on there being >=1 firing origin; the
                # eclipse/dwell are enforced per-origin above, so bypass the global gate.
                self._mech_ready = len(fire) > 0

        # If replication should be initiated, request subunits required for
        # building two replisomes per one origin of replication, and edit
        # access to oriC and chromosome domain attributes
        requests["bulk"] = []
        if self._should_initiate(n_oriC):
            requests["bulk"].append((self.replisome_trimers_idx, 6 * n_oriC))
            requests["bulk"].append((self.replisome_monomers_idx, 2 * n_oriC))

        # If there are no active forks return
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        if n_active_replisomes == 0:
            return

        # Get current locations of all replication forks
        (fork_coordinates,) = attrs(states["active_replisomes"], ["coordinates"])
        sequence_length = np.abs(np.repeat(fork_coordinates, 2))

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            len(self.sequences),
            self.basal_elongation_rate,
            states["timestep"],
        )

        sequences = buildSequences(
            self.sequences,
            np.tile(np.arange(4), n_active_replisomes // 2),
            sequence_length,
            self.elongation_rates,
        )

        # Count number of each dNTP in sequences for the next timestep
        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        # If one dNTP is limiting then limit the request for the other three by
        # the same ratio
        dNtpsTotal = counts(states["bulk"], self.dntps_idx)
        maxFractionalReactionLimit = (
            np.fmin(1, dNtpsTotal / sequenceComposition)
        ).min()

        # Request dNTPs
        requests["bulk"].append(
            (
                self.dntps_idx,
                (maxFractionalReactionLimit * sequenceComposition).astype(int),
            )
        )

        # requests no longer used in Step form; kept locally for clarity
        _ = requests

    def _evolve(self, states):
        timestep = states["timestep"]
        # Initialize the update dictionary
        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        # Module 1: Replication initiation
        # Get number of existing replisomes and oriCs
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        n_oriC = states["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC == 0:
            return update

        # Get attributes of existing chromosome domains
        domain_index_existing_domain, child_domains = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )

        # Which origins fire this tick. SYNCHRONOUS (default): all origins
        # (fire_mask all-True → byte-identical to the original logic). ASYNCHRONOUS
        # (DNAA_ASYNC_INITIATION): only the per-origin-saturated subset from _prepare.
        (domain_index_existing_oric,) = attrs(states["oriCs"], ["domain_index"])
        if self.async_initiation and self._fire_domains is not None:
            fire_mask = np.isin(domain_index_existing_oric, np.asarray(self._fire_domains))
        else:
            fire_mask = np.ones(n_oriC, dtype=bool)
        n_fire = int(fire_mask.sum())

        initiate_replication = False
        if n_fire > 0 and self._should_initiate(n_oriC):
            # Get number of available replisome subunits
            n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
            # Initiate replication only when
            # 1) The cell has reached the critical mass per oriC
            # 2) If mechanistic replisome option is on, there are enough
            # replisome subunits to assemble two replisomes per FIRING OriC.
            initiate_replication = not self.mechanistic_replisome or (
                np.all(n_replisome_trimers == 6 * n_fire)
                and np.all(n_replisome_monomers == 2 * n_fire)
            )

        # If all conditions are met, initiate a round of replication on the
        # firing origins (all of them in synchronous mode).
        if initiate_replication:
            self._last_init_time = self._cur_time  # global eclipse: stamp this initiation
            fire_oric_domains = domain_index_existing_oric[fire_mask]
            if self.async_initiation:
                for _d in fire_oric_domains.tolist():
                    self._last_init_time_by_domain[int(_d)] = self._cur_time

            # Get indexes of the domains that would be getting child domains
            # (domains that contain a FIRING origin)
            new_parent_domains = np.where(
                np.isin(domain_index_existing_domain, fire_oric_domains)
            )[0]

            # Calculate counts of new replisomes and domains to add
            n_new_replisome = 2 * n_fire
            n_new_domain = 2 * n_fire

            # Calculate the domain indexes of new domains and oriC's
            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_fire + 1, dtype=np.int32
            )

            # Reset the domain index of existing FIRING oriC's (non-firing keep
            # theirs), and add one new oriC per firing origin.
            new_oric_domain = np.asarray(domain_index_existing_oric).copy()
            new_oric_domain[fire_mask] = domain_index_new[:n_fire]
            update["oriCs"]["set"] = {"domain_index": new_oric_domain}
            update["oriCs"]["add"] = {
                "domain_index": domain_index_new[n_fire:],
            }

            # Add and set attributes of newly created replisomes.
            # New replisomes inherit the domain indexes of the oriC's they
            # were initiated from. Two replisomes are formed per firing oriC, one
            # on the right replichore, and one on the left.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_fire)
            right_replichore = right_replichore.tolist()
            domain_index_new_replisome = np.repeat(fire_oric_domains, 2)
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )
            update["active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            # Add and set attributes of new chromosome domains. All new domains
            # should have have no children domains.
            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
            )
            new_domains_update = {
                "add": {
                    "domain_index": domain_index_new,
                    "child_domains": new_child_domains,
                }
            }

            # Add new domains as children of the firing origins' domains
            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {"set": {"child_domains": child_domains}}
            update["chromosome_domains"].update(
                {**new_domains_update, **existing_domains_update}
            )

            # Decrement counts of replisome subunits (per firing origin)
            if self.mechanistic_replisome:
                update["bulk"].append((self.replisome_trimers_idx, -6 * n_fire))
                update["bulk"].append((self.replisome_monomers_idx, -2 * n_fire))

        # Write data from this module to a listener
        update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
            self.criticalMassPerOriC.magnitude
        )
        update["listeners"]["replication_data"]["critical_initiation_mass"] = (
            self.criticalInitiationMass.to(units.fg).magnitude
        )

        # Module 2: replication elongation
        # If no active replisomes are present, return immediately
        # Note: the new replication forks added in the previous module are not
        # elongated until the next timestep.
        if n_active_replisomes == 0:
            return update

        # Get allocated counts of dNTPs
        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        # Get attributes of existing replisomes
        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        # Build sequences to polymerize
        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

        sequences = buildSequences(
            self.sequences, sequence_indexes, sequence_length, self.elongation_rates
        )

        # Use polymerize algorithm to quickly calculate the number of
        # elongations each fork catalyzes
        reactionLimit = dNtpCounts.sum()

        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates,
        )

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        # Compute mass increase for each elongated sequence
        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.to(units.fg).magnitude,
        )

        # Compute masses that should be added to each replisome
        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        # Update positions of each fork
        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        # Reverse signs of fork coordinates on left replichore
        updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

        # Update attributes and submasses of replisomes
        (current_dna_mass,) = attrs(states["active_replisomes"], ["massDiff_DNA"])
        update["active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        # Update counts of polymerized metabolites
        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: replication termination
        # Determine if any forks have reached the end of their sequences. If
        # so, delete the replisomes and domains that were terminated.
        terminal_lengths = self.replichore_lengths[
            np.logical_not(right_replichore).astype(np.int64)
        ]
        terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

        # If any forks were terminated,
        if terminated_replisomes.sum() > 0:
            # Get domain indexes of terminated forks
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            # Get attributes of existing domains and full chromosomes
            (
                domain_index_domains,
                child_domains,
            ) = attrs(states["chromosome_domains"], ["domain_index", "child_domains"])
            (domain_index_full_chroms,) = attrs(
                states["full_chromosomes"], ["domain_index"]
            )

            # Initialize array of replisomes that should be deleted
            replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)

            # Count number of new full chromosomes that should be created
            n_new_chromosomes = 0

            # Initialize array for domain indexes of new full chromosomes
            domain_index_new_full_chroms = []

            for terminated_domain_index in terminated_domains:
                # Get all terminated replisomes in the terminated domain
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                # If both replisomes in the domain have terminated, we are
                # ready to split the chromosome and update the attributes.
                if terminated_domain_matching_replisomes.sum() == 2:
                    # Tag replisomes and domains with the given domain index
                    # for deletion
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index

                    # Get child domains of deleted domain
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    # Modify domain index of one existing full chromosome to
                    # index of first child domain
                    domain_index_full_chroms = domain_index_full_chroms.copy()
                    domain_index_full_chroms[
                        np.where(domain_index_full_chroms == terminated_domain_index)[0]
                    ] = child_domains_this_domain[0]

                    # Increment count of new full chromosome
                    n_new_chromosomes += 1

                    # Append chromosome index of new full chromosome
                    domain_index_new_full_chroms.append(child_domains_this_domain[1])

            # Delete terminated replisomes
            update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

            # Generate new full chromosome molecules
            if n_new_chromosomes > 0:
                chromosome_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_chroms,
                        "division_time": states["global_time"] + self.D_period,
                        "has_triggered_division": False,
                    }
                }

                # Reset domain index of existing chromosomes that have finished
                # replication
                chromosome_existing_update = {
                    "set": {"domain_index": domain_index_full_chroms}
                }

                update["full_chromosomes"].update(
                    {**chromosome_add_update, **chromosome_existing_update}
                )

            # Increment counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


def test_chromosome_replication():
    test_config = {}
    process = ChromosomeReplication(test_config)
    assert process is not None


if __name__ == "__main__":
    test_chromosome_replication()
