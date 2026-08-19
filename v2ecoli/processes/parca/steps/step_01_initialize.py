"""Step 1 — initialize (scatter).  Bootstrap the pipeline from raw KB data.

Takes the ``KnowledgeBaseEcoli`` loaded from the flat-file TSVs and builds
the nested ``SimulationDataEcoli`` object, then scatters its nine subsystem
dataclasses and twenty-odd top-level dicts onto their natural store paths
in the bigraph. No ``sim_data`` blob is passed downstream; every subsequent
step wires only the ports it needs.

Mathematical Model
------------------

Inputs:
- raw_data: a ``KnowledgeBaseEcoli`` instance holding every flat-file table
  (genes, RNAs, proteins, reactions, media, mass fractions, ...) parsed
  from ``v2ecoli.processes.parca/reconstruction/ecoli/flat/``. Delivered as a ``config``
  parameter (not a port wire) so bigraph-schema does not introspect the
  KB's deep internals at composite-construction time.

Parameters:
- debug (bool): forwarded to downstream steps; no effect on this one.

Calculation:
- sim_data = SimulationDataEcoli()
- sim_data.initialize(raw_data=raw_data, basal_expression_condition=...)
- split sim_data into its constituent subsystem objects and data dicts.

Outputs:
- Subsystem objects on ``process/*`` stores: transcription, translation,
  metabolism, rna_decay, complexation, equilibrium, two_component_system,
  transcription_regulation, replication.
- Top-level dataclass stores: mass, constants, growth_rate_parameters,
  adjustments, molecule_groups, molecule_ids, relation, getter, external_state.
- Pure-data dicts: conditions, condition_to_doubling_time, tf_to_fold_change,
  tf_to_active_inactive_conditions, condition_active_tfs, condition_inactive_tfs.
- bulk_molecules at ``internal_state/bulk_molecules``.
"""

import time
import warnings
from pathlib import Path

from process_bigraph import Step

from v2ecoli.processes.parca.reconstruction.ecoli.simulation_data import SimulationDataEcoli
from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle


def _is_valid_raw_data(obj) -> bool:
    """True iff ``obj`` is a usable ``KnowledgeBaseEcoli`` (exposes the flat-file
    tables ``sim_data.initialize`` reads). The first thing that method touches is
    ``raw_data.operons_on``, so that attribute is the cheapest liveness probe."""
    return hasattr(obj, "operons_on")


def _resolve_raw_data(config: dict):
    """Return a usable ``KnowledgeBaseEcoli`` for the injected ``config``.

    The registered ``parca`` composite is a *structural* document: it carries
    ``raw_data=None`` (see ``v2ecoli/composites/parca.py``), which the composite
    runtime materialises to an empty ``dict``. It was only ever runnable via the
    ``v2ecoli-parca`` CLI, which builds a real ``KnowledgeBaseEcoli`` and injects
    it. Driving the document through the workbench's generic runner instead fired
    this step with a plain ``dict`` → ``AttributeError: 'dict' object has no
    attribute 'operons_on'`` (vivarium-workbench #752).

    When a real KB is already injected (the CLI path) it is used unchanged. When
    it is missing (the workbench path) a real KB is loaded here — with production
    genotype defaults matching ``cli/parca.py`` (operons on; rRNA operons kept) —
    so the composite is self-sufficient. A declared ``bundle_manifest`` selects
    the ecoli-sources bundle; otherwise the default flat-file bundle is used.

    NOTE: this loads the KB and runs the FULL ParCa fit. Measured 2026-08-18 on a
    14-core laptop: ~4.6 min end to end (51 TF conditions). The log line below makes
    it explicit so a workbench run does not look hung; use the Runs-tab stop control
    to cancel. (Several repo docs still quote 4-8 h / ~300 conditions -- an inherited
    wcEcoli figure that does not describe this pipeline.)
    """
    raw_data = config.get("raw_data")
    if _is_valid_raw_data(raw_data):
        return raw_data

    manifest = config.get("bundle_manifest", "") or None
    # ``bundle_overrides`` was declared alongside ``bundle_manifest`` but never
    # read, so a study could name a private overlay and be silently ignored.
    # It is applied ON TOP of v2ecoli's own overrides, not instead of them
    # (SourceBundle chains them) — so naming one cannot revert v2ecoli's
    # diverged flat files.
    # Split on ';' -- symmetric with the CLI, which records its repeatable
    # --bundle-overrides as a ';'-joined string. Without this the whole joined
    # value is handed to SourceBundle as ONE path, which round-trips for a
    # single override and fails on two ("a.tsv;b.tsv" is not a file). The field
    # is provenance-only on the CLI path, so the breakage only appears where the
    # value is actually resolved -- which is exactly where it is hardest to
    # attribute back to how it was recorded.
    _raw_overrides = config.get("bundle_overrides", "") or ""
    overrides = [p for p in _raw_overrides.split(";") if p] or None
    # Likewise ``new_genes``: KnowledgeBaseEcoli has supported new-gene
    # insertion all along, but no entry point passed the option, so it was
    # unreachable and every build was "off". A private strain's new-gene flat
    # inputs arrive through the overlay above; this is what asks for them.
    new_genes = config.get("new_genes", "") or "off"
    print(
        "  Step 1: no KnowledgeBaseEcoli was injected (raw_data="
        f"{type(raw_data).__name__}); loading it now "
        f"(bundle_manifest={manifest or 'default'}, "
        f"bundle_overrides={overrides or 'none'}, new_genes={new_genes}). "
        "This runs the FULL ParCa "
        "fit — a few minutes; cancel from the Runs tab if unintended."
    )
    bundle = SourceBundle(base_manifest=manifest, overrides=overrides)
    return KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
        new_genes_option=new_genes,
        bundle=bundle,
    )


# Subsystem object outputs — each typed by its corresponding schema entry.
_SUBSYSTEM_PORTS = {
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'rna_decay':                'sim_data.rna_decay',
    'complexation':             'sim_data.complexation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    # Not every sim_data subsystem has a dedicated schema type registered;
    # use overwrite for the remainder.
    'adjustments':              'overwrite',
    'molecule_groups':          'overwrite',
    'molecule_ids':             'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'external_state':           'overwrite',
    # Escape-hatch port — the live SimulationDataEcoli instance.  Useful
    # for the handful of sub-functions that call methods defined on
    # sim_data itself (e.g. sim_data.calculate_ppgpp_expression) rather
    # than on one of its subsystems.  Mutations to the subsystems carried
    # by other ports propagate through this reference automatically.
    'sim_data_root':            'overwrite',
}

# Pure-data top-level dict outputs.
_DATA_LEAF_PORTS = {
    'tf_to_active_inactive_conditions': 'overwrite',
    'conditions':                       'overwrite',
    'condition_to_doubling_time':       'overwrite',
    'tf_to_fold_change':                'overwrite',
    'tf_to_direction':                  'overwrite',
    'condition_active_tfs':             'overwrite',
    'condition_inactive_tfs':           'overwrite',
    # Seed cell_specs as an empty dict — steps 3–9 populate per-condition
    # entries into it.  Tracked as its own leaf so every step's cell_specs
    # read/write is visible in the composite wires.
    'cell_specs':                       'overwrite',
    # Seeded empty; step 5 writes per-nutrient entries.
    'translation_supply_rate':          'overwrite',
    # Seeded empty; step 8 populates.
    'expected_dry_mass_increase_dict':  'overwrite',
    # Seeded empty; step 6 populates, step 7 consumes.
    'pPromoterBound':                   'overwrite',
    # sim_data.condition is a mutable runtime attr (default "basal")
    # used by create_bulk_container to pick the right nutrient media.
    'condition':                        'overwrite',
}

OUTPUT_PORTS = {
    'tick_1': 'overwrite',
    **_SUBSYSTEM_PORTS,
    **_DATA_LEAF_PORTS,
}


class InitializeStep(Step):
    """Run ``sim_data.initialize(raw_data=...)`` and scatter its subsystems."""

    description = (
        "Step 1 — initialize (scatter).\n\n"
        "Bootstraps the pipeline from the KnowledgeBaseEcoli flat-file tables:\n"
        "    sim_data = SimulationDataEcoli(); sim_data.initialize(raw_data)\n"
        "then scatters the nested object into its 9 subsystem dataclasses\n"
        "(transcription, translation, metabolism, rna_decay, complexation,\n"
        "equilibrium, two_component_system, transcription_regulation,\n"
        "replication) and ~20 top-level data dicts on their own store paths.\n"
        "No monolithic sim_data blob travels downstream — every later step\n"
        "wires only the ports it reads/writes."
    )

    config_schema = {
        'raw_data':                   'overwrite',
        'basal_expression_condition': {
            '_type': 'string',
            '_default': 'M9 Glucose minus AAs',
        },
        # Genotype identity. A ParCa build is identified by the ecoli-sources
        # bundle manifest its raw_data was built from, so these record WHICH
        # genome/dataset produced this fit. Declarative: the KnowledgeBase is
        # constructed by the runner and injected as `raw_data`, and these
        # fields do not build it. They exist so a study can name its genotype
        # in composite params, and so a mismatch between the declared genotype
        # and the injected one is caught rather than silently fitted.
        'bundle_manifest':  {'_type': 'string', '_default': ''},
        'bundle_overrides': {'_type': 'string', '_default': ''},
        # Name of a new_gene_data subdirectory to insert (e.g. a heterologous
        # pathway shipped by a private overlay bundle); '' / 'off' = none.
        # Unlike the two above this is NOT provenance-only — it changes the
        # genome the fit is built from.
        'new_genes':        {'_type': 'string', '_default': ''},
    }

    def _check_declared_genotype(self):
        """Warn when the declared bundle disagrees with the injected raw_data.

        Silent divergence here is the expensive failure: the fit succeeds and
        the resulting sim_data is attributed to a genotype it was not built
        from, which is exactly the provenance claim downstream studies rest on.
        """
        raw_data = self.config.get('raw_data')

        # new_genes is checked FIRST and separately: unlike the bundle fields it
        # is not merely provenance -- it changes the genome the fit is built
        # from. On the injected-raw_data path (the CLI, and any
        # build_parca_composite(raw_data=...) caller) this step never builds the
        # KB, so a config declaring an insertion against a wild-type KB would
        # otherwise fit WT, warn nothing, and record a genotype it does not have.
        declared_genes = self.config.get('new_genes', '') or 'off'
        actual_genes = getattr(raw_data, 'new_genes_option', None)
        if actual_genes is not None and declared_genes != actual_genes:
            warnings.warn(
                "ParCa genotype mismatch: step config declares new_genes "
                f"{declared_genes!r} but raw_data was built with "
                f"{actual_genes!r}. The fit proceeds against the INJECTED "
                "raw_data, so the resulting sim_data has the latter genome "
                "while its recorded config claims the former.",
                stacklevel=2,
            )

        declared = self.config.get('bundle_manifest', '')
        if not declared:
            return
        bundle = getattr(raw_data, '_bundle', None)
        actual = getattr(bundle, 'base_manifest', None)
        if actual is None:
            return
        if Path(declared).resolve() != Path(actual).resolve():
            warnings.warn(
                "ParCa genotype mismatch: step config declares bundle "
                f"manifest {declared!r} but raw_data was built from "
                f"{str(actual)!r}. The fit will proceed against the injected "
                "raw_data; the declared manifest is provenance only.",
                stacklevel=2,
            )

    def inputs(self):
        return {}

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()
        self._check_declared_genotype()
        raw_data = _resolve_raw_data(self.config)

        sim_data = SimulationDataEcoli()
        sim_data.initialize(
            raw_data=raw_data,
            basal_expression_condition=self.config.get(
                'basal_expression_condition', 'M9 Glucose minus AAs'),
        )

        # Scatter subsystems as live object references (no copies) so
        # downstream steps can mutate them in place and the mutations
        # persist in the store.
        out = {
            # subsystems
            'transcription':            sim_data.process.transcription,
            'translation':              sim_data.process.translation,
            'metabolism':               sim_data.process.metabolism,
            'rna_decay':                sim_data.process.rna_decay,
            'complexation':             sim_data.process.complexation,
            'equilibrium':              sim_data.process.equilibrium,
            'two_component_system':     sim_data.process.two_component_system,
            'transcription_regulation': sim_data.process.transcription_regulation,
            'replication':              sim_data.process.replication,
            'mass':                     sim_data.mass,
            'constants':                sim_data.constants,
            'growth_rate_parameters':   sim_data.growth_rate_parameters,
            'adjustments':              sim_data.adjustments,
            'molecule_groups':          sim_data.molecule_groups,
            'molecule_ids':             sim_data.molecule_ids,
            'relation':                 sim_data.relation,
            'getter':                   sim_data.getter,
            'bulk_molecules':           sim_data.internal_state.bulk_molecules,
            'external_state':           sim_data.external_state,
            'sim_data_root':            sim_data,
            # pure-data top-level dicts (copied — callers may mutate)
            'tf_to_active_inactive_conditions':
                dict(sim_data.tf_to_active_inactive_conditions),
            'conditions':                 dict(sim_data.conditions),
            'condition_to_doubling_time': dict(sim_data.condition_to_doubling_time),
            'tf_to_fold_change':          dict(sim_data.tf_to_fold_change),
            'tf_to_direction':            dict(sim_data.tf_to_direction),
            'condition_active_tfs':       dict(sim_data.condition_active_tfs),
            'condition_inactive_tfs':     dict(sim_data.condition_inactive_tfs),
            'cell_specs':                 {},
            'translation_supply_rate':    dict(sim_data.translation_supply_rate),
            'expected_dry_mass_increase_dict': {},
            'pPromoterBound':             {},
            'condition':                  sim_data.condition,
            'tick_1': True,
        }

        print(f"  Step 1 (initialize + scatter) completed in {time.time() - t0:.1f}s")
        return out
