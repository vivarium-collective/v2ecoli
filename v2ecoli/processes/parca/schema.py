"""
Bigraph-schema types for the v2ecoli.processes.parca store.

The store tree mirrors the structure of a fitted ``SimulationDataEcoli``.
Pure-data leaves (scalars, arrays, dicts) use the built-in ``overwrite``
type.  Subsystems that are live Python objects whose methods ParCa steps
invoke (e.g. ``sim_data.process.transcription`` is a ``Transcription``
instance with ``.set_ppgpp_expression(...)``) are registered here as named
types so the schema documents *what kind of object* lives at that path,
and so future dispatch (serialize, deep-copy, diff) can hook in per type.

All currently defined types are **overwrite-semantics subclasses**: apply
replaces the stored value.  This is the right default for the ParCa —
each step produces a new leaf value that replaces the previous one.
The named types are a forward-looking hook, not (yet) behaviorally
different from ``overwrite``.

The registry is exposed as ``register_parca_schema(core)`` which a
Composite builder calls on its core before constructing the spec.
"""

from dataclasses import dataclass

from bigraph_schema.schema import Overwrite


# ---------------------------------------------------------------------------
# Object-leaf type declarations — one per sim_data subsystem with behavior.
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class SimDataTranscription(Overwrite):
    """Opaque leaf holding a ``Transcription`` instance."""


@dataclass(kw_only=True)
class SimDataTranslation(Overwrite):
    """Opaque leaf holding a ``Translation`` instance."""


@dataclass(kw_only=True)
class SimDataMetabolism(Overwrite):
    """Opaque leaf holding a ``Metabolism`` instance."""


@dataclass(kw_only=True)
class SimDataMass(Overwrite):
    """Opaque leaf holding the ``Mass`` dataclass."""


@dataclass(kw_only=True)
class SimDataConstants(Overwrite):
    """Opaque leaf holding the ``Constants`` dataclass."""


@dataclass(kw_only=True)
class SimDataRnaDecay(Overwrite):
    """Opaque leaf holding an ``RnaDecay`` instance."""


@dataclass(kw_only=True)
class SimDataComplexation(Overwrite):
    """Opaque leaf holding a ``Complexation`` instance."""


@dataclass(kw_only=True)
class SimDataEquilibrium(Overwrite):
    """Opaque leaf holding an ``Equilibrium`` instance."""


@dataclass(kw_only=True)
class SimDataTwoComponentSystem(Overwrite):
    """Opaque leaf holding a ``TwoComponentSystem`` instance."""


@dataclass(kw_only=True)
class SimDataTranscriptionRegulation(Overwrite):
    """Opaque leaf holding a ``TranscriptionRegulation`` instance."""


@dataclass(kw_only=True)
class SimDataReplication(Overwrite):
    """Opaque leaf holding a ``Replication`` instance."""


@dataclass(kw_only=True)
class SimDataGrowthRateParameters(Overwrite):
    """Opaque leaf holding a ``GrowthRateParameters`` instance."""


# ---------------------------------------------------------------------------
# Pure-data leaf convenience aliases — also overwrite-semantics but named.
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class NumpyArray(Overwrite):
    """Leaf holding a numpy ndarray (any shape/dtype)."""


@dataclass(kw_only=True)
class StructuredArrayField(Overwrite):
    """Leaf holding a single field of a numpy structured array."""


@dataclass(kw_only=True)
class ConditionDict(Overwrite):
    """Leaf holding a dict keyed by condition label (e.g. 'basal', 'minimal_plus_AA')."""


PARCA_TYPES = {
    # object leaves
    'sim_data.transcription':            SimDataTranscription(),
    'sim_data.translation':              SimDataTranslation(),
    'sim_data.metabolism':               SimDataMetabolism(),
    'sim_data.mass':                     SimDataMass(),
    'sim_data.constants':                SimDataConstants(),
    'sim_data.rna_decay':                SimDataRnaDecay(),
    'sim_data.complexation':             SimDataComplexation(),
    'sim_data.equilibrium':              SimDataEquilibrium(),
    'sim_data.two_component_system':     SimDataTwoComponentSystem(),
    'sim_data.transcription_regulation': SimDataTranscriptionRegulation(),
    'sim_data.replication':              SimDataReplication(),
    'sim_data.growth_rate_parameters':   SimDataGrowthRateParameters(),
    # data-leaf aliases
    'numpy_array':                       NumpyArray(),
    'structured_array_field':            StructuredArrayField(),
    'condition_dict':                    ConditionDict(),
}


def register_parca_schema(core):
    """Register every v2ecoli.processes.parca-specific type on a process-bigraph core."""
    for name, entry in PARCA_TYPES.items():
        core.register_type(name, entry)
    return core
