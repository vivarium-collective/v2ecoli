"""Baseline whole-cell E. coli composite (55 processes, partitioned).

Upstream-parity architecture: the partitioned model matches the
vivarium-collective/vEcoli composite tick-for-tick. See AGENTS.md.

Migration note: the document-building body was migrated from
``v2ecoli/generate.py:build_document`` and
``v2ecoli/composite.py:_build_from_cache``.  Both legacy files were deleted
in Task 14.

Shared helpers (``make_edge``, ``inject_flow_dependencies``,
``_seed_state_from_defaults``, ``seed_mass_listener``,
``_normalize_boundary_units``, ``_make_instance``, ``_get_special_step``,
module-level constants) live in ``v2ecoli.composites._helpers``.
Architecture-specific helpers (``build_execution_layers``, ``DEFAULT_FEATURES``,
``_get_step_config``) are defined as private module-level functions here.
"""

from __future__ import annotations

import binascii
import copy
import os
from typing import Any

import numpy as np


def _derive_process_seed(master_seed: int, process_name: str) -> int:
    """Derive a per-process RNG seed from (master_seed, process_name).

    Without this, all stochastic processes inherit the same cache-derived seed,
    so multi-seed ensembles collapse to bit-identical trajectories. CRC32 with
    master_seed as the initial state, keyed by process name, gives each process
    its own 32-bit positive seed that varies across master_seeds (ensemble
    diversity) but is stable per (master_seed, process_name) pair (reproducibility).
    Mirrors the pattern already used in v2ecoli/steps/division.py.
    """
    return binascii.crc32(process_name.encode("utf-8"), master_seed) & 0x7FFFFFFF

from viva_superpowers.composite_generator import composite_generator, emitter_defaults

from v2ecoli.core import build_core, load_cache_bundle

# ---------------------------------------------------------------------------
# Shared helpers and constants
# ---------------------------------------------------------------------------
from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    _expand_flushes,
    set_default_emitter_decl,
    set_emitter_override,
    set_null_emitter_override,
    _find_workspace_root,
    CachedConfigLoader,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
    ALLOCATOR_LAYERS,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
# Re-export: baseline's biological store grouping (compartment -> molecular
# class), distilled from the retired `biological` composite. Pure annotation —
# baseline's EMITTED paths stay flat (global_time/bulk/listeners) so analyses are
# unaffected; display tools (Composite Explorer / loom, follow-up) read this to
# render the flat stores grouped as biology. See store_groups.py.
from v2ecoli.composites.store_groups import STORE_GROUPS as STORE_GROUPS  # noqa: PLC0414


# ---------------------------------------------------------------------------
# Execution layers (partitioned / baseline architecture)
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment (sequential sub-steps)
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone (no partitioning needed)
    ['ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation'], FLUSH,

    # NOTE: the dnaA-investigation mechanism steps — dnaa-3 (dnaa-box-binding +
    # dnaa_box_binding_listener), dnaa-4 (autoregulation in transcript_initiation),
    # and dnaa-5 (rida / ddah / dars + library/locus_copy_number) — remain in the
    # tree as DORMANT infrastructure but are NOT wired into the default baseline.
    # Main's default model is the pre-investigation WCM; these are activated only
    # by the dnaa-replication investigation (draft PR), which re-adds the layers.

    # Layer 3: TF binding

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: protein degradation (standalone — no resource competition)
    ['ecoli-protein-degradation'],

    # Layer 4b: standalone initiation/replication/complexation
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],
    # RNA degradation still partitioned (shares water with other processes)
    ['ecoli-rna-degradation_requester'],
    ['allocator_2'],
    ['ecoli-rna-degradation_evolver'], FLUSH,

    # Layer 5: partition layer 3 -- elongation requesters (parallel)
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    # Layer 5: partition layer 3 -- elongation evolvers (parallel)
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'], FLUSH,

    # Layer 6: chromosome structure + metabolism (sequential)
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # Layer 7: listeners (parallel)
    ['counts_deriver', 'ecoli-mass-listener',
     'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'], FLUSH,
    ['division'],
]

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
    # Opt-in runtime mass-conservation check. Runs after the mass listener
    # (dry_mass) and metabolism (environment.exchange). OFF by default: the
    # residual is not yet calibrated (net boundary exchange measures ~55x the
    # cell-mass change — see mass_conservation.py STATUS), so on a healthy run
    # it warns every tick. Enable per investigation via
    # enable_features('mass_conservation') before build_composite.
    'mass_conservation': {
        'insert_after': 'ecoli-mass-listener',
        'steps': ['ecoli-mass-conservation'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation + mass_conservation off by default

# Opt-in feature modules enabled by a caller for the NEXT build (the generator
# itself fixes the base set to DEFAULT_FEATURES). Mirrors the emitter-override
# pattern. Use enable_features('mass_conservation', ...) before build_composite.
_EXTRA_FEATURES: list = []


def enable_features(*names: str) -> None:
    """Enable opt-in feature modules (e.g. 'mass_conservation') for the next
    baseline build. Call before ``build_composite("ecoli_baseline", ...)``; pass no
    args to clear."""
    global _EXTRA_FEATURES
    _EXTRA_FEATURES = list(names)


def build_execution_layers(features=None):
    """Build EXECUTION_LAYERS from base + enabled feature modules."""
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
    for feat_name in (features or []):
        feat = FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        if 'insert_after' in feat:
            ref = feat['insert_after']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get('steps', []):
                        layers.insert(i + 1, [step_name])
                    break
        if 'insert_before' in feat:
            ref = feat['insert_before']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get('steps', [])):
                        layers.insert(i, [step_name])
                    break
        for listener in feat.get('listeners', []):
            for layer in layers:
                if isinstance(layer, list) and 'ecoli-mass-listener' in layer:
                    if listener not in layer:
                        layer.append(listener)
                    break
        # Replace an existing step name with another in-place (same layer/order).
        for old_name, new_name in (feat.get('replace') or {}).items():
            for layer in layers:
                if isinstance(layer, list):
                    layer[:] = [new_name if s == old_name else s for s in layer]
    return _expand_flushes(layers)


# Convenience re-export: ordered list of all step names in execution order.
FLOW_ORDER = [step for layer in build_execution_layers(DEFAULT_FEATURES) for step in layer]


# ---------------------------------------------------------------------------
# Step instantiation (partitioned / baseline architecture)
# ---------------------------------------------------------------------------

def _get_step_config(
    loader,
    step_name,
    core,
    process_cache=None,
    master_seed=0,
    transcript_initiation_mode: str | None = None,
    polypeptide_initiation_mode: str | None = None,
):
    """Get (instance, topology, edge_type[, in_topo, out_topo]) for a step.

    master_seed: investigation-level seed; per-process seeds are derived via
    _derive_process_seed(master_seed, base_name) so each stochastic process
    gets a distinct, reproducible seed across an ensemble.

    transcript_initiation_mode / polypeptide_initiation_mode: Phase-2 opt-in
    for the respective Process's jump-process kinetics. ``"discrete"``
    (default) → legacy multinomial sampling; ``"poisson"`` → per-target
    Poisson tau-leap. See the respective ``pdmp_initiation_mode`` config
    on each Process.
    """
    from v2ecoli.processes.equilibrium import Equilibrium
    from v2ecoli.processes.two_component_system import TwoComponentSystem
    from v2ecoli.processes.rna_maturation import RnaMaturation
    from v2ecoli.processes.complexation import Complexation
    from v2ecoli.steps.dnaa_box_binding import DnaABoxBinding
    from v2ecoli.steps.rida import Rida
    from v2ecoli.steps.ddah import Ddah
    from v2ecoli.steps.dars import Dars
    from v2ecoli.processes.protein_degradation import ProteinDegradation
    from v2ecoli.processes.rna_degradation import RnaDegradation
    from v2ecoli.processes.transcript_initiation import TranscriptInitiation
    from v2ecoli.processes.transcript_elongation import TranscriptElongation
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.processes.metabolism import Metabolism
    from v2ecoli.steps.partition import Requester, Evolver
    from v2ecoli.steps.derivers.mass_deriver import MassDeriver, PostDivisionMassDeriver
    from v2ecoli.steps.derivers.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.derivers.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.derivers.replication_data import ReplicationData
    from v2ecoli.steps.derivers.rnap_data import RnapData
    from v2ecoli.steps.derivers.translation_deriver import TranslationDeriver
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Handle allocators
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

    # Consolidated counts deriver: one step computing the RNA / monomer /
    # unique-molecule count readouts (byte-identical to the three former
    # listeners). Assemble its config from the three former config names.
    if step_name == 'counts_deriver':
        from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
        # Flat merge of the three former configs. Order matters: unique's
        # unique_ids (the one actually used) overwrites monomer's unused copy.
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

    # dnaa-3 Phase 2: DnaA-box binding step. No ParCa-generated config — built
    # from class defaults + cell_density / n_avogadro from the equilibrium
    # config + bulk_mass_data / submass_indices from tf_binding (used to
    # update DnaA_box.massDiff_* when DnaA moves bulk → bound).
    if step_name == 'dnaa-box-binding':
        try:
            eq_cfg = loader.get_config_by_name('ecoli-equilibrium') or {}
        except (KeyError, AttributeError):
            eq_cfg = {}
        try:
            tf_cfg = loader.get_config_by_name('ecoli-tf-binding') or {}
        except (KeyError, AttributeError):
            tf_cfg = {}
        dnaa_cfg = {
            'cell_density': eq_cfg.get('cell_density', 1100.0),
            'n_avogadro': eq_cfg.get('n_avogadro', 6.02214076e23),
            'seed': _derive_process_seed(master_seed, 'dnaa-box-binding'),
            'time_step': 1,
            'bulk_mass_data': tf_cfg.get('bulk_mass_data'),
            'bulk_molecule_ids': tf_cfg.get('bulk_molecule_ids'),
            'submass_indices': tf_cfg.get('submass_indices'),
        }
        instance = _make_instance(DnaABoxBinding, dnaa_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: RIDA — replisome-coupled DnaA-ATP inactivation. No ParCa config;
    # built from class defaults. rate_multiplier=0.0 gives the rida-knockout
    # variant (set via env RIDA_RATE_MULTIPLIER for the knockout sweep).
    if step_name == 'rida':
        rida_cfg = {
            'rate_multiplier': float(os.environ.get('RIDA_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'rida'),
            'time_step': 1,
        }
        instance = _make_instance(Rida, rida_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: DDAH — datA-locus-coupled DnaA-ATP hydrolysis.
    if step_name == 'ddah':
        ddah_cfg = {
            'rate_multiplier': float(os.environ.get('DDAH_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'ddah'),
            'time_step': 1,
        }
        instance = _make_instance(Ddah, ddah_cfg, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    # dnaa-5: DARS1/DARS2 — locus-copy-number-coupled DnaA reactivation.
    if step_name == 'dars':
        dars_cfg = {
            'dars1_multiplier': float(os.environ.get('DARS1_RATE_MULTIPLIER', '1.0')),
            'dars2_multiplier': float(os.environ.get('DARS2_RATE_MULTIPLIER', '1.0')),
            'seed': _derive_process_seed(master_seed, 'dars'),
            'time_step': 1,
        }
        instance = _make_instance(Dars, dars_cfg, core)
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

    # _instantiate_step inlined here (baseline/partitioned version)
    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
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
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    if isinstance(config, dict) and "seed" in config:
        config["seed"] = _derive_process_seed(master_seed, base_name)

    # Phase-2: thread {transcript,polypeptide}_initiation_mode overrides
    # into the ParCa-generated config so each Process's initialize()
    # picks up the jump-process mode.
    if (
        transcript_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-transcript-initiation'
    ):
        config["pdmp_initiation_mode"] = transcript_initiation_mode
    if (
        polypeptide_initiation_mode
        and isinstance(config, dict)
        and base_name == 'ecoli-polypeptide-initiation'
    ):
        config["pdmp_initiation_mode"] = polypeptide_initiation_mode

    # Partitioned processes: wrap with generic Requester/Evolver
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

    # Standalone steps
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


def _build_batch_document(
    core: Any,
    *,
    seed: int,
    n_seeds: int,
    n_generations: int,
    single_daughters: bool,
    time_step: float,
    max_duration: float,
    cache_dir: str,
    out_dir: str,
    experiment_id: str,
    emitter: str,
    analyses: Any,
    study: str,
    parallel: str,
    variants: dict | None,
    knockouts: list[str] | None,
    config_overrides: dict | None,
    media: str,
) -> dict:
    """Build the batch-orchestrator document (seeds × generations lineage).

    ``baseline`` dispatches here when ``n_seeds>1`` or ``n_generations>1``.
    Returns a cheap one-step document whose ``BatchBaselineRunner`` fans out one
    baseline lineage per seed at RUN time (Ray, sequential fallback), emitting to
    a shared hive-partitioned parquet sweep (+ per-lineage zarr) and then running
    the post-simulation analysis flush. Absorbs the former ``batch_baseline``
    composite; every per-seed cell is built by ``baseline`` itself.

    Parameter mapping: baseline's ``seed`` is the lineage base seed (seeds are
    ``seed .. seed+n_seeds-1``); ``knockouts`` + ``config_overrides`` fold into
    the runner's panel-wide ``base_config_overrides`` (applied to every seed);
    ``media`` threads through to each per-seed ``baseline`` build.
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.perturbations import translation_efficiency_override
    from v2ecoli.steps.batch_baseline_runner import (
        BatchBaselineRunner, DEFAULT_EMITTER)
    from v2ecoli.composites._helpers import _make_instance, make_edge

    # Emitter reconciliation: the single-cell sinks 'sqlite'/'null' are
    # meaningless for a multi-lineage sweep. A batch accepts parquet | xarray |
    # both. (baseline's default 'parquet' is valid here; pass 'both' for the
    # per-lineage zarr the dashboard per-run charts read.)
    batch_emitter = emitter or "parquet"
    if batch_emitter in ("sqlite", "null"):
        raise ValueError(
            f"emitter={emitter!r} is single-cell only; a batch run "
            "(n_seeds>1 or n_generations>1) accepts 'parquet', 'xarray', or "
            "'both'.")

    # Knockouts + config_overrides -> panel-wide base_config_overrides, applied to
    # EVERY seed's lineage (not one variant arm). Resolve knockouts once against
    # the cache; a caller config_override key wins on a clash.
    base_config_overrides: dict = {}
    if knockouts:
        bundle = load_cache_bundle(cache_dir)
        base_config_overrides = translation_efficiency_override(
            bundle, list(knockouts))
    if config_overrides:
        base_config_overrides = {**base_config_overrides, **config_overrides}

    runner_config = {
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "base_seed": int(seed),
        "single_daughters": bool(single_daughters),
        "time_step": float(time_step),
        "max_duration": float(max_duration),
        "variants": dict(variants or {}),
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "experiment_id": experiment_id,
        "emitter": batch_emitter or DEFAULT_EMITTER,
        "analyses": analyses,
        "study": study,
        "parallel": parallel or "",
        "base_config_overrides": base_config_overrides,
        "media": media,
    }
    runner = _make_instance(BatchBaselineRunner, runner_config, core)
    state = {
        "batch": {},  # empty; the runner writes per-seed results here at run time
        "global_time": 0.0,
        "batch_runner": make_edge(
            runner, BatchBaselineRunner.topology, edge_type="step",
            config=runner_config),
    }
    return {"state": state}


@composite_generator(
    name="ecoli_baseline",
    description="55-process partitioned whole-cell E. coli model — upstream-parity architecture",
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "transcript_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for transcription initiation. "
                "'discrete' (default, legacy): multinomial event distribution "
                "with exact Σ N_i = n_target. 'poisson': per-promoter "
                "Poisson(n_target · p_i) tau-leap."
            ),
        },
        "polypeptide_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "Phase-2 jump-process opt-in for translation initiation. "
                "Same dispatch as transcript_initiation_mode but for "
                "PolypeptideInitiation; ribosome activation per protein "
                "becomes per-protein Poisson(n_target · p_i) tau-leap "
                "instead of one global multinomial draw."
            ),
        },
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": "Declarative '<process>.<key>': value config overrides (variants)",
        },
        "knockouts": {
            "type": "list",
            "default": [],
            "description": "Genes to knock out at the translation level — EcoCyc "
                           "gene ids (EG10526) or monomer ids (LACY-MONOMER[c]). "
                           "Each named gene's translation efficiency is zeroed on "
                           "the cached polypeptide-initiation config, so no protein "
                           "is made — a functional knockout with no ParCa re-fit. "
                           "Empty = plain baseline. See v2ecoli.perturbations.",
        },
        "media": {
            "type": "string",
            "default": "minimal",
            "description": "Initial growth medium — any condition in the cache's "
                           "saved_media (e.g. 'minimal_plus_amino_acids', "
                           "'minimal_succinate', 'minimal_minus_oxygen'). Sets the "
                           "environment's initial media_id so media_update shifts "
                           "the cell onto that condition on the first tick and "
                           "metabolism responds (e.g. amino-acid-rich media grows "
                           "faster) — a lightweight media perturbation from the "
                           "existing cache, no ParCa re-fit. Default 'minimal' = "
                           "unchanged. For a rigorously-calibrated condition, run a "
                           "per-condition ParCa cache instead (see showcase-4).",
        },
        "features": {
            "type": "list",
            "default": [],
            "description": "Opt-in feature-module names to insert in addition to "
                           "the boolean toggles (e.g. ['mass_conservation']). "
                           "Each must be a key in FEATURE_MODULES.",
        },
        # --- Biological feature toggles (insert/remove feature-module steps) ---
        "ppgpp_regulation": {
            "type": "bool",
            "default": True,
            "description": "Enable ppGpp-mediated regulation of transcription "
                           "initiation (ppgpp-initiation step). On by default.",
        },
        "trna_attenuation": {
            "type": "bool",
            "default": False,
            "description": "Enable tRNA transcriptional attenuation "
                           "(trna-attenuation-config step). Off by default.",
        },
        "supercoiling": {
            "type": "bool",
            "default": False,
            "description": "Enable DNA supercoiling dynamics (dna-supercoiling-step "
                           "+ dna_supercoiling_listener). Off by default.",
        },
        "mass_conservation": {
            "type": "bool",
            "default": False,
            "description": "Enable the opt-in runtime mass-conservation check "
                           "(ecoli-mass-conservation step). Off by default — the "
                           "residual is not yet calibrated, so it warns each tick.",
        },
        # --- Observation sink selection ---
        "emitter": {
            "type": "string",
            "choices": ["parquet", "sqlite", "xarray", "null", "both"],
            "default": "parquet",
            "description": "Observation sink. Single-cell: parquet (hive "
                           "column store, default), sqlite (time-series db), "
                           "xarray (in-memory arrays), or null (global_time only). "
                           "Batch (n_seeds>1 / n_generations>1): parquet, xarray "
                           "(per-lineage zarr), or 'both' (parquet + zarr; what "
                           "the dashboard per-run charts read). 'both' is batch "
                           "only; 'sqlite'/'null' are single-cell only.",
        },
        "injected_processes": {
            "type": "map",
            "default": {},
            "description": "Fork process-injection spec "
                           "{fork_repo, add_processes, swap_processes, "
                           "process_configs, topology, time_step}; empty = none.",
        },
        # --- Batch / lineage knobs (absorbed from the former batch_baseline) ----
        # n_seeds>1 OR n_generations>1 switches baseline from a single 55-process
        # cell document to a one-step batch-orchestrator document that fans out
        # one baseline lineage per seed at RUN time (Ray, sequential fallback),
        # emits to a shared sweep, then runs the post-sim analysis flush. The
        # single-cell default (n_seeds=1, n_generations=1) is unchanged.
        "n_seeds": {
            "type": "integer",
            "default": 1,
            "description": "Number of independent seeds to run (vEcoli's "
                           "n_init_sims); seeds are seed .. seed+n_seeds-1. 1 = a "
                           "single cell (default). >1 launches a batch run.",
        },
        "n_generations": {
            "type": "integer",
            "default": 1,
            "description": "Cell-division generations to follow per seed lineage. "
                           ">1 (or n_seeds>1) launches a batch run.",
        },
        "single_daughters": {
            "type": "bool",
            "default": True,
            "description": "Batch runs only: follow ONE daughter per division "
                           "(vEcoli's default). Off = binary-tree lineage.",
        },
        "time_step": {
            "type": "float",
            "default": 1.0,
            "description": "Batch runs only: simulation time step in seconds.",
        },
        "max_duration": {
            "type": "float",
            "default": 3600.0,
            "description": "Batch runs only: per-generation sim-time cap (seconds).",
        },
        "variants": {
            "type": "map",
            "default": {},
            "description": "Batch runs only: vEcoli-style variant grid "
                           "({name: {target, value}}) crossed with the seed range.",
        },
        "out_dir": {
            "type": "string",
            "default": "",
            "description": "Batch runs only: output root for the parquet sweep + "
                           "zarr stores. Empty = this run's own dir under the "
                           "workbench, else out/batch_baseline.",
        },
        "experiment_id": {
            "type": "string",
            "default": "baseline",
            "description": "Batch runs only: id stamped into the parquet "
                           "partitions and zarr store names.",
        },
        "analyses": {
            "type": "string",
            "choices": ["applicable", "none"],
            "default": "applicable",
            "description": "Batch runs only: 'applicable' runs every ported "
                           "analysis at the scales this batch covers; 'none' skips.",
        },
        "study": {
            "type": "string",
            "default": "",
            "description": "Batch runs only: owning study slug for the flush's "
                           "outputs. Empty = infer from out_dir.",
        },
        "parallel": {
            "type": "string",
            "default": "ray",
            "description": "Batch runs only: 'ray' to fan out across worker "
                           "processes; '' for sequential.",
        },
    },
    default_n_steps=2700,
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    emitters=[
        {
            # Default observation sink for standalone builds: a vEcoli-shaped
            # ParquetEmitter (hive-partitioned, column-oriented — captures the
            # raw bulk count array + listeners that downstream/vEcoli-parity
            # analyses need). out_dir is omitted on purpose: the emitter step
            # resolves it to <workspace>/.pbg/parquet-runs. External overrides
            # (set_parquet_emitter_override / set_emitter_override) still win.
            "address": "local:ParquetEmitter",
            "config": {},
            "paths": ["global_time", "bulk", "listeners"],
        },
    ],
)
def baseline(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    transcript_initiation_mode: str = "discrete",
    polypeptide_initiation_mode: str = "discrete",
    config_overrides: dict | None = None,
    knockouts: list[str] | None = None,
    media: str = "minimal",
    features: list | None = None,
    ppgpp_regulation: bool = True,
    trna_attenuation: bool = False,
    supercoiling: bool = False,
    mass_conservation: bool = False,
    emitter: str = "parquet",
    bundle: dict | None = None,
    injected_processes: dict | None = None,
    n_seeds: int = 1,
    n_generations: int = 1,
    single_daughters: bool = True,
    time_step: float = 1.0,
    max_duration: float = 3600.0,
    variants: dict | None = None,
    out_dir: str = "",
    experiment_id: str = "baseline",
    analyses: Any = "applicable",
    study: str = "",
    parallel: str = "ray",
) -> dict:
    """Build the process-bigraph state document for the baseline architecture.

    Migrated from ``v2ecoli/generate.py:build_document`` +
    ``v2ecoli/composite.py:_build_from_cache``.  Returns a plain dict
    suitable for ``Composite(doc, core=core)``; does NOT wrap in Composite.

    The biological feature set is assembled from the boolean toggles
    (``ppgpp_regulation`` on by default; the rest off) plus any opt-in
    modules a caller registered via :func:`enable_features` (back-compat).

    Args:
        core: bigraph-schema core.  If None, one is created via build_core().
        seed: Random seed for stochastic initialisation.
        cache_dir: Path to the ParCa cache directory (must contain
            ``initial_state.json`` and ``sim_data_cache.dill``).
        transcript_initiation_mode: Phase-2 opt-in for the PDMP transcript
            initiation dispatch — ``discrete`` (default) or the piecewise-
            deterministic mode.
        polypeptide_initiation_mode: same dispatch as
            ``transcript_initiation_mode`` but for polypeptide initiation.
        config_overrides: declarative '<process>.<key>': value config patches
            (variants), applied on top of any knockout patch.
        knockouts: genes to knock out at the translation level (EcoCyc gene ids
            or monomer ids). Each gene's translation efficiency is zeroed on the
            cached polypeptide-initiation config — a functional knockout with no
            ParCa re-fit. Empty/None = plain baseline.
        media: initial growth medium — any condition in the cache's saved_media.
            Sets the environment's initial media_id so media_update shifts the
            cell onto that condition on the first tick and metabolism responds
            (lightweight media perturbation from the existing cache, no ParCa
            re-fit). 'minimal' (default) leaves it unchanged.
        n_seeds, n_generations: >1 on either switches baseline to a BATCH run —
            a one-step orchestrator document that fans out one baseline lineage
            per seed (seeds seed..seed+n_seeds-1) at run time and flushes the
            ported analyses (absorbs the former batch_baseline composite). The
            other batch knobs (single_daughters, time_step, max_duration,
            variants, out_dir, experiment_id, analyses, study, parallel) apply
            only in batch mode; knockouts/media/config_overrides carry through to
            every seed. n_seeds==1, n_generations==1 (default) = single cell.
        ppgpp_regulation: insert the ppGpp-regulation feature module (default on).
        trna_attenuation: insert the tRNA-attenuation feature module (default off).
        supercoiling: insert the DNA-supercoiling feature module (default off).
        mass_conservation: insert the mass-conservation check (default off).
        emitter: observation sink for the internal 'emitter' step — one of
            ``parquet`` (default), ``sqlite``, ``xarray``, ``null``.
        bundle: optional pre-loaded cache bundle (as returned by
            ``load_cache_bundle``). When given, the cache is not re-read from
            ``cache_dir`` — lets callers building many composites from the same
            cache (ensembles, sweeps) load it once and reuse it.

    Returns:
        Process-bigraph document dict with keys ``state``,
        ``skip_initial_steps``, ``sequential_steps``, ``flow_order``.
    """
    if core is None:
        core = build_core()

    # Batch dispatch: n_seeds>1 or n_generations>1 turns baseline from a single
    # 55-process cell into a one-step batch-orchestrator document (absorbs the
    # former batch_baseline composite). The single-cell path below is untouched
    # for n_seeds==1, n_generations==1 (bit-identical to plain baseline).
    if int(n_seeds) > 1 or int(n_generations) > 1:
        return _build_batch_document(
            core, seed=seed, n_seeds=n_seeds, n_generations=n_generations,
            single_daughters=single_daughters, time_step=time_step,
            max_duration=max_duration, cache_dir=cache_dir, out_dir=out_dir,
            experiment_id=experiment_id, emitter=emitter, analyses=analyses,
            study=study, parallel=parallel, variants=variants,
            knockouts=knockouts, config_overrides=config_overrides, media=media)

    if bundle is None:
        bundle = load_cache_bundle(cache_dir)

    # Translation-level gene knockouts (design PR #341, folded in from the former
    # KO_baseline composite). Resolve the knockouts against this same cache bundle
    # into a `translation_efficiencies` config override and merge it UNDER any
    # caller config_overrides — an explicit override key wins on a clash (they are
    # being deliberate). An unknown/non-coding gene raises here, at build time,
    # not silently mid-run. Empty knockouts = plain baseline (no cache touch).
    if knockouts:
        from v2ecoli.perturbations import translation_efficiency_override
        ko = translation_efficiency_override(bundle, list(knockouts))
        config_overrides = {**ko, **(config_overrides or {})}

    # Deep-copy initial_state: a reused bundle (e.g. one load_cache_bundle()
    # shared across many baseline() calls, as in parameter sweeps / UQ ensembles)
    # otherwise hands every composite the SAME initial_state arrays. v2ecoli's
    # in-place bulk arrays then mutate that shared state during run(), so each
    # subsequent build resumes from the previous run's advanced state (mass
    # accumulates across samples, eventually triggering a spurious mid-run
    # division). configs is already deep-copied below for the same reason —
    # initial_state needs the same isolation.
    initial_state = copy.deepcopy(bundle["initial_state"])
    configs = bundle["configs"]
    if config_overrides:
        # Deep-copy before patching: load_cache_bundle returns the cache dict
        # by reference (lru_cache-shared); mutating it would corrupt other runs.
        configs = copy.deepcopy(configs)
        for path, value in config_overrides.items():
            proc, _, key = path.partition(".")
            if not key:
                raise ValueError(f"override path {path!r} must be '<process>.<key>'.")
            if "." in key:
                raise ValueError(
                    f"override path {path!r}: nested keys not supported; "
                    "use '<process>.<top-level-key>'.")
            if proc not in configs:
                raise KeyError(f"override target process {proc!r} not in cache configs.")
            configs[proc][key] = value
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    # Assemble the feature set from the explicit boolean toggles. Order follows
    # FEATURE_MODULES so insertions are deterministic. The legacy
    # _EXTRA_FEATURES / enable_features() global is still honoured (back-compat
    # for callers like scripts/pr_session_report.py and the mass-conservation
    # behavior test) and unions in any modules it requested.
    _toggle_features = {
        'ppgpp_regulation': ppgpp_regulation,
        'trna_attenuation': trna_attenuation,
        'supercoiling': supercoiling,
        'mass_conservation': mass_conservation,
    }
    _requested_features = list(features or [])
    features = [name for name, on in _toggle_features.items() if on]
    for f in _EXTRA_FEATURES:
        if f not in features:
            features.append(f)
    # Explicit per-call opt-in feature modules (e.g. mass_conservation).
    for f in _requested_features:
        if f not in features:
            features.append(f)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    # Media perturbation (from the existing cache — no ParCa re-fit). The cache's
    # initial environment is 'minimal'; the media_update step swaps in a different
    # condition's external concentrations on the first tick when the environment's
    # media_id differs from its own (config-seeded) current id. So a media change
    # is just: set the initial environment.media_id to a saved_media condition.
    # Validate here so a typo'd condition fails at build time, not silently.
    if media and media != cell_state.get('environment', {}).get('media_id'):
        _saved = (configs.get('media_update') or {}).get('saved_media') or {}
        if media not in _saved:
            raise ValueError(
                f"media={media!r} is not a condition in the cache's saved_media. "
                f"Available: {sorted(_saved)}")
        cell_state.setdefault('environment', {})['media_id'] = media

    # Pre-create virtual stores
    for store in ['listeners', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    # Pre-create feature module stores
    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })

    # Initialize next_update_time for all partitioned processes
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    # Pre-create shared request/allocate/process stores
    cell_state.setdefault('request', {})
    cell_state.setdefault('allocate', {})
    for proc_name in ALL_PARTITIONED:
        cell_state['request'].setdefault(proc_name, {'bulk': {}})
        cell_state['allocate'].setdefault(proc_name, {'bulk': {}})
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Mock loader: serves cache configs + a minimal sim_data stand-in (see
    # CachedConfigLoader in _helpers — replaces the old nested _CachedLoader).
    loader = CachedConfigLoader(
        configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    # Build execution layers for the requested feature set
    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    # Resolve the 'emitter' param to the right override / declared-default so
    # the internal 'emitter' step materialises the chosen sink. The selection
    # only adjusts this generator's OWN scoped knobs (the declared default decl
    # + the sqlite/null overrides it sets here); it is restored in the finally
    # so nothing leaks into a later composite built in the same process. Any
    # EXTERNAL override a caller set before this build (set_parquet_emitter_override
    # / set_emitter_override / set_null_emitter_override) still wins, because
    # _get_special_step checks those external overrides before the declared
    # default and we only set our own override when none is already active.
    import v2ecoli.composites._helpers as _h  # noqa: PLC0415

    if emitter not in ("parquet", "sqlite", "xarray", "null"):
        raise ValueError(
            f"emitter={emitter!r} not recognised; expected one of "
            "parquet, sqlite, xarray, null.")

    _emitter_decls = emitter_defaults(baseline)
    _default_decl = _emitter_decls[0] if _emitter_decls else None

    # Snapshot external overrides so we can detect 'caller already pinned one'
    # and restore them exactly on exit.
    _ext_parquet = _h._PARQUET_EMITTER_OVERRIDE
    _ext_sqlite = _h._EMITTER_OVERRIDE
    _ext_null = _h._NULL_EMITTER_OVERRIDE
    _any_external = (_ext_parquet is not None or _ext_sqlite is not None
                     or _ext_null)

    set_default_emitter_decl(_default_decl)

    if emitter == "xarray" and not _any_external:
        # XArray is emitted OUT OF BAND by the workflow/lineage runner: its
        # transducer + view describe per-composite variable shapes that are only
        # knowable lazily on the first populated emit tick (see
        # workflow/lineage.py:_emit_xarray), so there is no self-contained
        # in-document XArrayEmitter step. We therefore mirror the canonical
        # xarray contract here: minimise the INTERNAL 'emitter' step to
        # global_time only (set_null_emitter_override) and let the external
        # XArray sink own persistence. Selecting 'xarray' in a plain
        # build_composite/dashboard run thus behaves like 'null' internally;
        # the real XArray output appears when run under the lineage workflow.
        import warnings
        warnings.warn(
            "emitter='xarray': the internal emitter is minimised to global_time "
            "only; real XArray persistence is produced out-of-band by the "
            "lineage workflow runner (v2ecoli.workflow.lineage), not by this "
            "in-document emitter step.")
        set_null_emitter_override(True)
    elif emitter == "sqlite" and not _any_external:
        # Minimal persistent SQLite sink. Resolve the workspace-shared DB (the
        # dashboard's Simulations-DB tab aggregates from it); fall back to out/.
        _ws_root = _find_workspace_root()
        _sqlite_dir = (str(_ws_root / ".pbg") if _ws_root is not None
                       else "out")
        set_emitter_override({
            "file_path": _sqlite_dir,
            "db_file": "composite-runs.db",
        })
    elif emitter == "null" and not _any_external:
        set_null_emitter_override(True)
    # emitter == "parquet": the declared parquet default (set above) is used.

    _process_cache = {}
    try:
        for step_name in flow_order:
            config = _get_step_config(
                loader, step_name, core, _process_cache, master_seed=seed,
                transcript_initiation_mode=transcript_initiation_mode,
                polypeptide_initiation_mode=polypeptide_initiation_mode,
            )
            if config is not None:
                if len(config) == 5:
                    instance, topology, edge_type, in_topo, out_topo = config
                    cell_state[step_name] = make_edge(
                        instance, topology, input_topology=in_topo,
                        output_topology=out_topo, edge_type=edge_type)
                else:
                    instance, topology, edge_type = config
                    cell_state[step_name] = make_edge(
                        instance, topology, edge_type=edge_type)
    finally:
        set_default_emitter_decl(None)
        # Restore external overrides to exactly their pre-build values (we only
        # ever changed them when none was active, so this clears ours).
        set_emitter_override(_ext_sqlite)
        set_null_emitter_override(_ext_null)

    # Place shared PartitionedProcess instances in the process store
    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    seed_mass_listener(cell_state, core)

    # Shape step (Skalnik et al. 2023): derive the capsule cell geometry from
    # mass — length from volume = mass/density, fixed width. Reads the whole
    # listeners.mass sub-store, writes the top-level 'shape' store. Added as a
    # FINAL execution layer (after the mass listener) so inject_flow_dependencies
    # wires it into the per-tick flow: it recomputes length/volume from the
    # current cell_mass every step, so the envelope tracks growth over the sim.
    # (The 'shape' store is seeded with all keys: a map[float] store only merges
    # onto existing keys.)
    if core is not None:
        from v2ecoli.cell_shape import ShapeStep, zero_shape
        core.register_link("ShapeStep", ShapeStep)
        cell_state['shape'] = zero_shape()
        cell_state['shape_step'] = {
            '_type': 'step',
            'address': 'local:ShapeStep',
            'config': {'width_um': 1.0, 'density_g_per_ml': 1.1,
                       'periplasm_fraction': 0.2},
            'inputs': {'mass': ['listeners', 'mass']},
            'outputs': {'shape': ['shape']},
        }
        execution_layers = execution_layers + [['shape_step']]
        flow_order = [step for layer in execution_layers for step in layer]

    inject_flow_dependencies(
        cell_state, flow_order, layers=execution_layers)

    if injected_processes and (
            injected_processes.get("add_processes")
            or injected_processes.get("swap_processes")
            or injected_processes.get("exclude_processes")):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "..", "scripts"))
        from scripts._compare.inject import (
            resolve_injections, apply_injected_processes, remove_processes)
        # Add half: convert + inject the new processes (add_processes plus the
        # TARGETS of swap_processes). resolve_injections needs the fork repo only
        # when there is something to add.
        if (injected_processes.get("add_processes")
                or injected_processes.get("swap_processes")):
            specs = resolve_injections(injected_processes["fork_repo"],
                                       injected_processes)
            apply_injected_processes(cell_state, flow_order, core, specs)
        # Remove half: drop the swapped-out SOURCES and any exclude_processes, so
        # a swap is a true replace (not a co-existing add).
        remove_processes(cell_state, flow_order,
                         list((injected_processes.get("swap_processes") or {}).keys()))
        remove_processes(cell_state, flow_order,
                         list(injected_processes.get("exclude_processes") or []))

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': flow_order,
    }
