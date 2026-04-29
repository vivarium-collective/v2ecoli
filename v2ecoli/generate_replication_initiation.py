"""
Document generation for v2ecoli (replication-initiation architecture).

This architecture is the home for a more biologically detailed model of
chromosome-replication initiation. The curated source for the new biology
is ``docs/references/replication_initiation.md`` (with the underlying
PDF at ``docs/references/replication_initiation_molecular_info.pdf``);
the structured constants live in
``v2ecoli.data.replication_initiation.molecular_reference``.

At the time of this scaffolding PR ``build_replication_initiation_document``
is a thin pass-through to the baseline ``generate.build_document``: the
composite is functionally identical to ``baseline``. Subsequent PRs add
divergent wiring here — see the draft-PR plan for the phased order:

  1. Explicit DnaA-ATP / DnaA-ADP molecular species in the bulk schema.
  2. oriC binding-state model (R1/R2/R4 high-affinity occupancy +
     low-affinity DnaA-ATP filament + IHF) gating initiation, replacing
     the mass threshold in ``chromosome_replication.ChromosomeReplication``.
  3. SeqA sequestration on hemimethylated GATC sites (~10 min window).
  4. RIDA: DnaN (β-clamp) + Hda → DnaA-ATP hydrolysis on replicating DNA.
  5. DDAH: IHF + datA-mediated DnaA-ATP hydrolysis (backup to RIDA).
  6. DARS1 / DARS2: ADP-DnaA → apo-DnaA → ATP-DnaA reactivation.
  7. dnaA promoter autoregulation by DnaA-ATP / DnaA-ADP occupancy.
"""

import copy

import numpy as np

from v2ecoli.generate import (
    DEFAULT_FEATURES,
    FLOW_ORDER,
    build_document as _baseline_build_document,
    make_edge,
)
from v2ecoli.processes.chromosome_replication_dnaA_gated import (
    DnaAGatedChromosomeReplication,
)
from v2ecoli.processes.dars import DARS, NAME as DARS_NAME, TOPOLOGY as DARS_TOPOLOGY
from v2ecoli.processes.dnaA_box_binding import (
    DnaABoxBinding, NAME as BINDING_NAME, TOPOLOGY as BINDING_TOPOLOGY,
)
from v2ecoli.processes.rida import RIDA, NAME as RIDA_NAME, TOPOLOGY as RIDA_TOPOLOGY


_CHROMOSOME_REPLICATION_STEP_NAME = 'ecoli-chromosome-replication'

# SeqA sequestration window — ~10 min in fast-growth E. coli per the
# curated reference (Katayama et al. 2017, in the curated PDF).
SEQA_SEQUESTRATION_WINDOW_S: float = 600.0


def _swap_in_dnaA_gated_chromosome_replication(document, *,
                                                seqA_window_s: float = 0.0):
    """Replace the baseline ChromosomeReplication instance with the
    DnaA-gated subclass. Keeps the same step name, topology, and
    config — only the underlying class differs. The subclass calls
    super()._prepare() so existing bookkeeping (bulk indices,
    replisome subunit accounting) is preserved.

    When ``seqA_window_s > 0``, the subclass also enforces a
    refractory window after each initiation event, modeling SeqA
    sequestration of the newly-replicated origin (Phase 4)."""
    cell_state = document['state']['agents']['0']
    edge = cell_state.get(_CHROMOSOME_REPLICATION_STEP_NAME)
    if not isinstance(edge, dict) or 'instance' not in edge:
        return document
    old_instance = edge['instance']
    raw_config = getattr(old_instance, '_raw_config',
                         getattr(old_instance, 'parameters', {}))
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    if seqA_window_s > 0:
        config['seqA_sequestration_window_s'] = seqA_window_s
    new_instance = DnaAGatedChromosomeReplication(config)
    edge['instance'] = new_instance
    cls = type(new_instance)
    edge['address'] = f'local:{cls.__module__}.{cls.__qualname__}'
    return document


# Identifier used by reports / network views to label this architecture.
ARCHITECTURE_NAME = 'replication_initiation'


# Equilibrium reactions that the new architecture neutralizes so that
# RIDA's output (DnaA-ADP) and DARS's output (regenerated DnaA-ATP)
# can accumulate kinetically rather than being instantly re-equilibrated
# by mass-action against cellular ATP / ADP. The reactions remain in the
# config (so the equilibrium SS solver still runs successfully) but
# their stoichMatrix columns are zeroed, which kills their effect on
# molecule counts.
DEACTIVATED_EQUILIBRIUM_REACTIONS: tuple[str, ...] = (
    'MONOMER0-4565_RXN',  # apo-DnaA + ADP <-> DnaA-ADP
)


def _deactivate_equilibrium_reactions(document, reactions):
    """Zero the stoichMatrix columns for the named equilibrium reactions
    on the live equilibrium step instance. The mass-action drive for
    those reactions is gone; molecule counts on either side are no
    longer coupled by the equilibrium step."""
    cell_state = document['state']['agents']['0']
    eq_edge = cell_state.get('ecoli-equilibrium')
    if not isinstance(eq_edge, dict) or 'instance' not in eq_edge:
        return document
    instance = eq_edge['instance']
    rxn_ids = list(getattr(instance, 'reaction_ids', []))
    sm = getattr(instance, 'stoichMatrix', None)
    if sm is None or not rxn_ids:
        return document
    sm = np.array(sm, copy=True)
    deactivated = []
    for name in reactions:
        if name in rxn_ids:
            sm[:, rxn_ids.index(name)] = 0
            deactivated.append(name)
    instance.stoichMatrix = sm
    instance.product_indices = [
        idx for idx in np.where(np.any(sm > 0, axis=1))[0]]
    instance._deactivated_reactions = tuple(deactivated)
    return document


def _splice_rida(document, seed):
    """Add the RIDA step to the cell_state and append it to flow_order.

    The reaction-level definition (``RXN0-7444`` in the metabolism FBA)
    exists upstream but carries zero flux; this dedicated step actively
    transfers DnaA-ATP -> DnaA-ADP at a rate proportional to active
    replisome count. It runs after every other step in the flow so it
    sees the post-equilibrium DnaA-ATP pool each tick.
    """
    cell_state = document['state']['agents']['0']
    if RIDA_NAME in cell_state:
        return document

    rida_config = {
        'time_step': 1.0,
        'seed': int(seed),
    }
    instance = RIDA(rida_config)
    cell_state[RIDA_NAME] = make_edge(
        instance, RIDA_TOPOLOGY, edge_type='step', config=rida_config)

    document.setdefault('flow_order', [])
    if RIDA_NAME not in document['flow_order']:
        document['flow_order'] = list(document['flow_order']) + [RIDA_NAME]
    return document


def _splice_dnaA_box_binding(document, seed):
    """Add the DnaABoxBinding step to the cell_state and append it to
    flow_order. The step reads the DnaA-ATP/ADP bulk pools and writes
    DnaA_bound on each box per a per-region equilibrium occupancy
    sample. This is the first thing in the model that ever sets
    DnaA_bound to True.
    """
    cell_state = document['state']['agents']['0']
    if BINDING_NAME in cell_state:
        return document
    binding_config = {'time_step': 1.0, 'seed': int(seed)}
    instance = DnaABoxBinding(binding_config)
    cell_state[BINDING_NAME] = make_edge(
        instance, BINDING_TOPOLOGY, edge_type='step', config=binding_config)
    document.setdefault('flow_order', [])
    if BINDING_NAME not in document['flow_order']:
        document['flow_order'] = list(document['flow_order']) + [BINDING_NAME]
    return document


def _splice_dars(document, seed):
    """Add the DARS step to the cell_state and append it to flow_order.

    DARS releases ADP from DnaA-ADP, regenerating apo-DnaA. Paired
    with the still-active ``MONOMER0-160_RXN`` equilibrium that
    re-loads apo-DnaA with ATP, this closes the DnaA nucleotide cycle:

        DnaA-ATP --[RIDA, gated by replisomes]--> DnaA-ADP
        DnaA-ADP --[DARS]-->                       apo-DnaA
        apo-DnaA + ATP <==[equilibrium]==>         DnaA-ATP

    With both Phase 5 (RIDA) and Phase 7 (DARS) wired, the DnaA-ATP
    fraction reaches a steady state inside the literature 30-70% band
    instead of monotonically depleting.
    """
    cell_state = document['state']['agents']['0']
    if DARS_NAME in cell_state:
        return document

    dars_config = {
        'time_step': 1.0,
        'seed': int(seed),
    }
    instance = DARS(dars_config)
    cell_state[DARS_NAME] = make_edge(
        instance, DARS_TOPOLOGY, edge_type='step', config=dars_config)

    document.setdefault('flow_order', [])
    if DARS_NAME not in document['flow_order']:
        document['flow_order'] = list(document['flow_order']) + [DARS_NAME]
    return document


def build_replication_initiation_document(
    initial_state,
    configs,
    unique_names,
    dry_mass_inc_dict=None,
    core=None,
    seed=0,
    features=None,
    enable_rida: bool = True,
    enable_dars: bool = True,
    enable_dnaA_box_binding: bool = True,
    enable_dnaA_gated_initiation: bool = True,
    enable_seqA_sequestration: bool = True,
):
    """Build the replication-initiation document.

    Starts from the baseline document and splices in the divergent
    processes / config edits that distinguish this architecture. The
    feature flags let callers (e.g. the per-phase report) build any
    cumulative slice of the architecture for direct comparison:

      * ``enable_rida=False, enable_dars=False`` → identical to baseline.
      * ``enable_rida=True,  enable_dars=False`` → Phase 5 only.
        DnaA-ATP starts depleting because there is no reactivation.
      * ``enable_rida=True,  enable_dars=True``  → current full state.
        RIDA + DARS form a closed cycle around DnaA-ATP / DnaA-ADP.
      * ``enable_rida=False, enable_dars=True``  → DARS without anything
        producing DnaA-ADP for it to consume; nothing happens. Allowed
        but not biologically meaningful.

    The DnaA-ADP equilibrium override is bound to ``enable_rida`` —
    without RIDA there is no DnaA-ADP to protect from mass-action.
    """
    document = _baseline_build_document(
        initial_state=copy.deepcopy(initial_state),
        configs=configs,
        unique_names=unique_names,
        dry_mass_inc_dict=dry_mass_inc_dict or {},
        core=core,
        seed=seed,
        features=features,
    )
    if enable_rida:
        document = _splice_rida(document, seed=seed)
        document = _deactivate_equilibrium_reactions(
            document, DEACTIVATED_EQUILIBRIUM_REACTIONS)
    if enable_dars:
        document = _splice_dars(document, seed=seed)
    if enable_dnaA_box_binding:
        document = _splice_dnaA_box_binding(document, seed=seed)
    if enable_dnaA_gated_initiation:
        seqA_window = (SEQA_SEQUESTRATION_WINDOW_S
                       if enable_seqA_sequestration else 0.0)
        document = _swap_in_dnaA_gated_chromosome_replication(
            document, seqA_window_s=seqA_window)
    return document
