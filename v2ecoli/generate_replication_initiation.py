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

# Re-export the FLOW_ORDER and DEFAULT_FEATURES from baseline so callers
# (e.g. ``reports/workflow_report.py``) can keep importing them through
# this module if they want the replication-initiation architecture's
# layering, even when the layering itself has not yet diverged.
from v2ecoli.generate import (
    FLOW_ORDER,
    DEFAULT_FEATURES,
    build_document as _baseline_build_document,
)


# Identifier used by reports / network views to label this architecture.
ARCHITECTURE_NAME = 'replication_initiation'


def build_replication_initiation_document(
    initial_state,
    configs,
    unique_names,
    dry_mass_inc_dict=None,
    core=None,
    seed=0,
    features=None,
):
    """Build the replication-initiation document.

    Currently delegates to the baseline document builder. Divergence from
    baseline will be implemented here as the new processes land.
    """
    return _baseline_build_document(
        initial_state=copy.deepcopy(initial_state),
        configs=configs,
        unique_names=unique_names,
        dry_mass_inc_dict=dry_mass_inc_dict or {},
        core=core,
        seed=seed,
        features=features,
    )
