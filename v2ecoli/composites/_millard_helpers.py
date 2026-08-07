"""Shared Millard-composite helpers.

Single source for the symbols that the Millard-family composites
(``baseline_millard`` — both its ``lqr=False`` and ``lqr=True`` paths — and
``reactor_bird_coupled_millard``) all need:

- ``_get_step_config`` — the per-step config dispatcher (identical to
  ``baseline.py``'s but with ``ecoli-metabolism`` intentionally ABSENT,
  since these composites replace tFBA metabolism with the Millard ODE).
  It still handles the LQR / FBA-bridge / ref-growth / millard-bulk-indexer
  step names for the ``lqr=True`` path; the env-driven ``lqr=False`` path
  simply never requests those features.
- ``_register_millard_pdmp_links`` — ``core_extensions`` hook registering the
  Millard / FBA-bridge / multi-state-LQR Process classes.
- ``FEATURE_MODULES`` / ``DEFAULT_FEATURES`` — opt-in feature-module table.
- ``MILLARD_EDGES`` / ``MILLARD_BULK_INDEXER`` / ``REF_GROWTH_DRIVER`` — step
  name constants used by the dispatcher / feature table.
- ``_derive_process_seed`` — per-process deterministic seed derivation.

Extracted from ``baseline_millard`` so the keeper composites don't import
private symbols out of one another.
"""

from __future__ import annotations

import binascii

from v2ecoli.composites._helpers import (
    _make_instance,
    _get_special_step,
    ALLOCATOR_LAYERS,
    ALL_PARTITIONED,
    PARTITIONED_PROCESSES,
)


def _derive_process_seed(master_seed: int, process_name: str) -> int:
    return binascii.crc32(process_name.encode("utf-8"), master_seed) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Step name constants
# ---------------------------------------------------------------------------

MILLARD_EDGES = ['millard-pdmp-metabolism', 'lqr-controller']

# Step that consumes FBABridge's parallel-dict output and writes deltas to
# the structured bulk store — kept as a fallback if the inline-bulk-emit
# path in MillardPDMPMetabolism is ever disabled.
MILLARD_BULK_INDEXER = 'millard-bulk-indexer'

# Optional teleonomic growth driver — closes the W₂ gap exposed by
# scripts/compare_pdmp_vs_phase0.py while a proper biomass-flux Process
# (task #21 full form) is built. Off unless `ref_growth_driver` feature
# is enabled in the composite call.
REF_GROWTH_DRIVER = 'ref-growth-driver'


# ---------------------------------------------------------------------------
# Feature-module table
# ---------------------------------------------------------------------------

FEATURE_MODULES = {
    'supercoiling': {
        'insert_after': 'ecoli-chromosome-structure',
        'steps': ['dna-supercoiling-step'],
        'listeners': ['dna_supercoiling_listener'],
    },
    'ppgpp_regulation': {
        'insert_before': 'ecoli-transcript-initiation',
        'steps': ['ppgpp-initiation'],
    },
    'trna_attenuation': {
        'insert_before': 'ecoli-transcript-elongation_requester',
        'steps': ['trna-attenuation-config'],
    },
    # Teleonomic Phase-0 reference growth driver. Inserts after the
    # metabolism Process so each tick's order is: Millard → bulk indexer
    # (delta-mode) → ref-growth-driver (scale precursors at μ_ref).
    # Off by default; opt-in via composite features=['ref_growth_driver'].
    'ref_growth_driver': {
        'insert_after': 'millard-pdmp-metabolism',
        'steps': [REF_GROWTH_DRIVER],
    },
    # Opt-in runtime mass-conservation check (shared with baseline.py). Runs
    # after the mass listener; reads environment.exchange (now populated by the
    # Millard edge's medium-exchange accounting) + listeners.mass. Off by
    # default; enable via features=['mass_conservation'].
    'mass_conservation': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['ecoli-mass-conservation'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']


# ---------------------------------------------------------------------------
# Step config dispatch — identical to baseline but with metabolism removed
# from the standalone-step registry.
# ---------------------------------------------------------------------------

def _get_step_config(
    loader,
    step_name,
    core,
    process_cache=None,
    master_seed=0,
    ref_growth_flux_source: str | None = None,
    ref_growth_feedback_tau_s: float | None = None,
    ref_growth_feedback_period_ticks: int | None = None,
    transcript_initiation_mode: str | None = None,
    transcript_init_prob_scale: float | None = None,
    polypeptide_init_prob_scale: float | None = None,
    polypeptide_initiation_mode: str | None = None,
):
    from v2ecoli.processes.equilibrium import Equilibrium
    from v2ecoli.processes.two_component_system import TwoComponentSystem
    from v2ecoli.processes.rna_maturation import RnaMaturation
    from v2ecoli.processes.complexation import Complexation
    from v2ecoli.processes.protein_degradation import ProteinDegradation
    from v2ecoli.processes.transcript_initiation import TranscriptInitiation
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.steps.partition import Requester, Evolver
    from v2ecoli.steps.derivers.mass_deriver import MassDeriver, PostDivisionMassDeriver
    from v2ecoli.steps.derivers.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.derivers.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.derivers.replication_data import ReplicationData
    from v2ecoli.steps.derivers.rnap_data import RnapData
    from v2ecoli.steps.derivers.translation_deriver import TranslationDeriver
    from v2ecoli.steps.derivers.likelihood_collector import LikelihoodCollector
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Allocators
    if step_name.startswith('allocator'):
        from v2ecoli.steps.allocator import Allocator
        try:
            alloc_config = loader.get_config_by_name('allocator')
        except Exception:
            alloc_config = {}
        layer_procs = ALLOCATOR_LAYERS.get(step_name, ALL_PARTITIONED)
        if not alloc_config.get('process_names'):
            alloc_config['process_names'] = ALL_PARTITIONED
        alloc_config['layer_processes'] = layer_procs
        instance = _make_instance(Allocator, alloc_config, core)
        topo = instance.topology
        return instance, topo, 'step'

    # Millard-PDMP edges are handled separately (in baseline body), so the
    # config dispatcher never sees them. Return None to skip.
    if step_name in MILLARD_EDGES or step_name in (
            'millard-with-lqr', 'fba-bridge'):
        return None

    # ref-growth-driver: Phase-0 reference growth driver.
    # flux_source overrides come through master_seed → flux_source extras.
    if step_name == REF_GROWTH_DRIVER:
        from v2ecoli.steps.ref_growth_driver import RefGrowthDriver
        driver_config = {
            "seed": _derive_process_seed(master_seed, step_name),
        }
        flux_source = ref_growth_flux_source
        if flux_source:
            driver_config["flux_source"] = flux_source
        if ref_growth_feedback_tau_s is not None:
            driver_config["feedback_tau_s"] = float(ref_growth_feedback_tau_s)
        if ref_growth_feedback_period_ticks is not None:
            driver_config["feedback_period_ticks"] = int(
                ref_growth_feedback_period_ticks)
        instance = _make_instance(
            RefGrowthDriver,
            driver_config,
            core,
        )
        # Reads bulk + listeners.mass (read-only, for closed-loop WATER[c]
        # density regulation); writes only bulk.
        in_topo = {"bulk": ("bulk",), "listeners_mass": ("listeners", "mass")}
        out_topo = {"bulk": ("bulk",)}
        return instance, instance.topology, 'step', in_topo, out_topo

    # millard-bulk-indexer: instantiate from its own module; no sim-data
    # config required. Pass explicit input/output topologies — the
    # indexer READS central_metabolites and WRITES bulk; the default
    # make_edge behavior of reusing topology for both directions would
    # declare it as a central_metabolites writer too, conflicting with
    # millard-with-lqr and silently dropping the latter's update.
    # Phase-3 likelihood collector: no ParCa config, no special-step
    # registration. Instantiate directly with the minimal tick config so
    # we don't fall through to loader.get_config_by_name (which raises
    # → _get_special_step → silently returns None and the step is
    # dropped from the composite).
    if step_name == 'likelihood_collector':
        from v2ecoli.steps.derivers.likelihood_collector import LikelihoodCollector
        instance = _make_instance(LikelihoodCollector, {}, core)
        topo = getattr(instance, 'topology', {})
        if callable(topo):
            topo = topo()
        return instance, topo, 'step'

    if step_name == MILLARD_BULK_INDEXER:
        from v2ecoli.steps.millard_bulk_indexer import MillardBulkIndexer
        instance = _make_instance(
            MillardBulkIndexer,
            {"seed": _derive_process_seed(master_seed, step_name)},
            core,
        )
        in_topo = {
            "bulk": ("bulk",),
            "cm_view": ("shared", "central_metabolites"),
        }
        out_topo = {"bulk": ("bulk",)}
        return instance, instance.topology, 'step', in_topo, out_topo

    # Consolidated counts deriver: one step computing the RNA / monomer /
    # unique-molecule count readouts (byte-identical to the three former
    # listeners). Assemble its config from the three former config names.
    if step_name == 'counts_deriver':
        from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
        merged_cfg = {}
        for cfg_name in ('RNA_counts_listener', 'monomer_counts_listener',
                         'unique_molecule_counts'):
            try:
                merged_cfg.update(loader.get_config_by_name(cfg_name) or {})
            except (KeyError, AttributeError):
                pass
        instance = _make_instance(CountsDeriver, merged_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'


    try:
        config = loader.get_config_by_name(base_name)
    except (KeyError, AttributeError):
        try:
            config = loader.get_config_by_name(step_name)
        except (KeyError, AttributeError):
            return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    # STANDALONE_STEPS — note: ecoli-metabolism is INTENTIONALLY ABSENT.
    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-complexation': Complexation,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-chromosome-replication': ChromosomeReplication,
    }

    SIMPLE_STEPS = {
        'ecoli-mass-listener': MassDeriver,
        'post-division-mass-listener': PostDivisionMassDeriver,
        'rna_synth_prob_listener': RnaSynthProb,
        'dna_supercoiling_listener': DnaSupercoiling,
        'replication_data_listener': ReplicationData,
        'rnap_data_listener': RnapData,
        'ribosome_data_listener': TranslationDeriver,
        'likelihood_collector': LikelihoodCollector,
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    if isinstance(config, dict) and "seed" in config:
        config["seed"] = _derive_process_seed(master_seed, base_name)

    # Phase-2 jump-process opt-ins. Merge mode overrides into the ParCa-
    # generated config so each Process's initialize() picks them up.
    if (
        transcript_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["pdmp_initiation_mode"] = transcript_initiation_mode
    if (
        transcript_init_prob_scale is not None
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["transcript_init_prob_scale"] = float(
            transcript_init_prob_scale)
    if (
        polypeptide_init_prob_scale is not None
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["polypeptide_init_prob_scale"] = float(
            polypeptide_init_prob_scale)
    if (
        polypeptide_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["pdmp_initiation_mode"] = polypeptide_initiation_mode

    if base_name in PARTITIONED_PROCESSES:
        proc_cls = PARTITIONED_PROCESSES[base_name]
        if base_name in process_cache:
            process = process_cache[base_name]
        else:
            from v2ecoli.library.ecoli_step import set_current_core
            set_current_core(core)
            process = proc_cls(config)
            set_current_core(None)
            process_cache[base_name] = process
        topology = dict(config.get('topology', {}) or {})
        if not topology:
            topology = getattr(process, 'topology',
                               getattr(proc_cls, 'topology', {}))
            if callable(topology):
                topology = topology()
            topology = dict(topology)

        if step_name.endswith('_requester'):
            instance = Requester({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            if 'request' in out_ports:
                out_topo['request'] = ('request', base_name)
            if 'listeners' in out_ports:
                out_topo['listeners'] = topology.get('listeners', ('listeners',))
            return instance, topology, 'step', in_topo, out_topo

        elif step_name.endswith('_evolver'):
            instance = Evolver({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['allocate'] = ('allocate', base_name)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            for port in out_ports:
                if port in ('next_update_time', 'process', 'allocate',
                            'global_time', 'timestep'):
                    continue
                if port in topology:
                    out_topo[port] = topology[port]
                elif port == 'listeners':
                    out_topo['listeners'] = ('listeners',)
            return instance, topology, 'step', in_topo, out_topo

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = _make_instance(step_cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = _make_instance(cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    return None


def _register_millard_pdmp_links(core):
    """core_extensions hook — register the three Millard-PDMP Process classes."""
    from v2ecoli.steps.millard_with_lqr import register as register_millard
    from v2ecoli.steps.fba_bridge import register as register_bridge
    from v2ecoli.steps.lqr_controller_multistate import register as register_lqr_ms
    register_millard(core)
    register_bridge(core)
    register_lqr_ms(core)
    return core
